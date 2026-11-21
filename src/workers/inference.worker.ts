import * as ort from 'onnxruntime-web/webgpu';
import { BallDetection, WorkerInMessage, WorkerOutMessage } from '../types';

const workerScope = self as unknown as {
  postMessage: (msg: WorkerOutMessage) => void;
  onmessage: ((ev: MessageEvent<WorkerInMessage>) => void) | null;
};

let session: ort.InferenceSession | null = null;
let inputSize = 320;
let confThreshold = 0.22;
let preserveAspectRatio = false;
let offscreenCanvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let inputBuffer: Float32Array | null = null;

async function initModel(
  modelUrl: string,
  wasmPaths: string,
  size = 320,
  threshold = 0.22
) {
  inputSize = size;
  confThreshold = threshold;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = wasmPaths;

  offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
  ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  inputBuffer = new Float32Array(3 * inputSize * inputSize);

  let modelBytes: Uint8Array | null = null;
  try {
    if ('caches' in self) {
      const cached = await caches.match(modelUrl);
      if (cached && cached.ok) {
        modelBytes = new Uint8Array(await cached.arrayBuffer());
      }
    }
    if (!modelBytes) {
      const resp = await fetch(modelUrl);
      if (resp.ok) {
        modelBytes = new Uint8Array(await resp.arrayBuffer());
      }
    }
  } catch {
    modelBytes = null;
  }

  const createSession = (opts: ort.InferenceSession.SessionOptions) =>
    modelBytes
      ? ort.InferenceSession.create(modelBytes, opts)
      : ort.InferenceSession.create(modelUrl, opts);

  let activeBackend = 'WASM';
  if ('gpu' in navigator) {
    try {
      session = await createSession({
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'basic',
      });
      activeBackend = 'WebGPU';
    } catch {
      // Browser, adapter or model does not support WebGPU.
    }
  }
  if (!session) {
    session = await createSession({
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'basic',
    });
  }
  const output = session.outputMetadata[0];
  if (!output?.isTensor || output.shape.length !== 3 ||
      (output.shape[1] !== 5 && output.shape[1] !== 84)) {
    throw new Error('Expected a single-class baseball or 80-class COCO YOLO model');
  }
  // Use the legacy stretch for COCO and training-style padding for the baseball model.
  preserveAspectRatio = output.shape[1] === 5;
  workerScope.postMessage({ type: 'READY', backend: activeBackend });
}

function preprocess(bitmap: ImageBitmap) {
  if (!ctx || !inputBuffer) throw new Error('Canvas buffer not initialized');
  const scale = Math.min(inputSize / bitmap.width, inputSize / bitmap.height);
  const drawWidth = preserveAspectRatio ? Math.max(1, Math.round(bitmap.width * scale)) : inputSize;
  const drawHeight = preserveAspectRatio ? Math.max(1, Math.round(bitmap.height * scale)) : inputSize;
  const padX = Math.floor((inputSize - drawWidth) / 2);
  const padY = Math.floor((inputSize - drawHeight) / 2);
  if (preserveAspectRatio) {
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, inputSize, inputSize);
  }
  ctx.drawImage(bitmap, padX, padY, drawWidth, drawHeight);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
  const area = inputSize * inputSize;
  const inv255 = 1 / 255;

  for (let i = 0; i < area; i++) {
    const px = i * 4;
    const r = data[px];
    const g = data[px + 1];
    const b = data[px + 2];
    inputBuffer[i] = r * inv255;
    inputBuffer[i + area] = g * inv255;
    inputBuffer[i + 2 * area] = b * inv255;
  }
  return {
    tensor: new ort.Tensor('float32', inputBuffer, [1, 3, inputSize, inputSize]),
    padX,
    padY,
    drawWidth,
    drawHeight,
  };
}

function parseBestBall(
  output: ort.Tensor,
  origWidth: number,
  origHeight: number,
  timestampMs: number,
  { padX, padY, drawWidth, drawHeight }: ReturnType<typeof preprocess>
): BallDetection | null {
  const data = output.data as Float32Array;
  const channels = output.dims[1];
  const anchors = output.dims[2];
  const numClasses = channels - 4;
  const targetClassIdx = numClasses === 1 ? 0 : 32; // 自訓單類別=0, COCO預訓練=32(sports ball)
  if (targetClassIdx >= numClasses) return null;

  const offset = (4 + targetClassIdx) * anchors;
  let bestScore = confThreshold;
  let bestIdx = -1;

  for (let i = 0; i < anchors; i++) {
    const score = data[offset + i];
    const x = data[i];
    const y = data[anchors + i];
    if (
      score > bestScore &&
      x >= padX && x < padX + drawWidth &&
      y >= padY && y < padY + drawHeight &&
      data[2 * anchors + i] > 0 && data[3 * anchors + i] > 0
    ) {
      bestScore = score;
      bestIdx = i;
    }
  }
  if (bestIdx === -1) return null;

  const scaleX = origWidth / drawWidth;
  const scaleY = origHeight / drawHeight;
  return {
    x: (data[bestIdx] - padX) * scaleX,
    y: (data[anchors + bestIdx] - padY) * scaleY,
    width: data[2 * anchors + bestIdx] * scaleX,
    height: data[3 * anchors + bestIdx] * scaleY,
    confidence: bestScore,
    timestampMs,
  };
}

workerScope.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  try {
    if (msg.type === 'INIT') {
      await initModel(msg.modelUrl, msg.wasmPaths, msg.inputSize, msg.confThreshold);
    } else if (msg.type === 'INFER') {
      const start = performance.now();
      try {
        if (!session) throw new Error('ONNX model is not ready');
        const frame = preprocess(msg.bitmap);
        const results = await session.run({ [session.inputNames[0]]: frame.tensor });
        const detection = parseBestBall(
          results[session.outputNames[0]],
          msg.origWidth,
          msg.origHeight,
          msg.timestampMs,
          frame,
        );
        workerScope.postMessage({
          type: 'RESULT',
          detection,
          inferTimeMs: Number((performance.now() - start).toFixed(1)),
        });
      } finally {
        msg.bitmap.close();
      }
    }
  } catch (err) {
    workerScope.postMessage({
      type: 'ERROR',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
