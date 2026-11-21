import * as ort from 'onnxruntime-web';
import { BallDetection, WorkerInMessage, WorkerOutMessage } from '../types';

const workerScope = self as unknown as {
  postMessage: (msg: WorkerOutMessage) => void;
  onmessage: ((ev: MessageEvent<WorkerInMessage>) => void) | null;
};

let session: ort.InferenceSession | null = null;
let inputSize = 320;
let confThreshold = 0.35;
let offscreenCanvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let inputBuffer: Float32Array | null = null;

async function initModel(
  modelUrl: string,
  wasmPaths: string,
  size = 320,
  threshold = 0.35
) {
  inputSize = size;
  confThreshold = threshold;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = wasmPaths;

  offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
  ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  inputBuffer = new Float32Array(3 * inputSize * inputSize);

  let activeBackend = 'wasm';
  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    });
    activeBackend = 'webgpu/wasm';
  } catch {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }
  workerScope.postMessage({ type: 'READY', backend: activeBackend });
}

function preprocess(bitmap: ImageBitmap): ort.Tensor {
  if (!ctx || !inputBuffer) throw new Error('Canvas buffer not initialized');
  ctx.drawImage(bitmap, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
  const area = inputSize * inputSize;
  const inv255 = 1 / 255;

  for (let i = 0; i < area; i++) {
    const px = i * 4;
    inputBuffer[i] = data[px] * inv255;
    inputBuffer[i + area] = data[px + 1] * inv255;
    inputBuffer[i + 2 * area] = data[px + 2] * inv255;
  }
  return new ort.Tensor('float32', inputBuffer, [1, 3, inputSize, inputSize]);
}

function parseBestBall(
  output: ort.Tensor,
  origWidth: number,
  origHeight: number,
  timestampMs: number
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
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  if (bestIdx === -1) return null;

  const scaleX = origWidth / inputSize;
  const scaleY = origHeight / inputSize;
  return {
    x: data[0 * anchors + bestIdx] * scaleX,
    y: data[1 * anchors + bestIdx] * scaleY,
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
        if (!session) return;
        const tensor = preprocess(msg.bitmap);
        const results = await session.run({ [session.inputNames[0]]: tensor });
        const detection = parseBestBall(
          results[session.outputNames[0]],
          msg.origWidth,
          msg.origHeight,
          msg.timestampMs
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
