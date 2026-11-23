import * as ort from 'onnxruntime-web/wasm';
import { BallDetection, PitchActionPhase, WorkerInMessage, WorkerOutMessage } from '../types';

const workerScope = self as unknown as {
  postMessage: (msg: WorkerOutMessage) => void;
  onmessage: ((ev: MessageEvent<WorkerInMessage>) => void) | null;
};

let session: ort.InferenceSession | null = null;
let actionSession: ort.InferenceSession | null = null;
let inputSize = 320;
const ACTION_SIZE = 160;
const ACTION_LABELS: PitchActionPhase[] = ['idle', 'pitching_end', 'pitching_start'];
let confThreshold = 0.22;
let preserveAspectRatio = false;
let offscreenCanvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let inputBuffer: Float32Array | null = null;
let actionCanvas: OffscreenCanvas | null = null;
let actionCtx: OffscreenCanvasRenderingContext2D | null = null;
let actionBuffer: Float32Array | null = null;

async function fetchModelBytes(url: string): Promise<Uint8Array | null> {
  try {
    if ('caches' in self) {
      const cached = await caches.match(url);
      if (cached && cached.ok) return new Uint8Array(await cached.arrayBuffer());
    }
    const resp = await fetch(url);
    if (resp.ok) return new Uint8Array(await resp.arrayBuffer());
  } catch {
    // Fall back to URL loading.
  }
  return null;
}

async function initModel(
  modelUrl: string,
  wasmPaths: string,
  size = 320,
  threshold = 0.22,
  actionModelUrl?: string
) {
  inputSize = size;
  confThreshold = threshold;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = wasmPaths;

  offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
  ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  inputBuffer = new Float32Array(3 * inputSize * inputSize);

  // Use WASM directly for fast, reliable startup across Chrome/Safari/Intel GPU (avoids WebGPU shader-compile hangs).
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
  });
  const activeBackend = 'WASM';

  // Auto-sync inputSize to the loaded ONNX model's actual tensor shape so cached/upgraded models never mismatch.
  const inputMeta = session.inputMetadata[0];
  if (inputMeta?.isTensor && typeof inputMeta.shape?.[2] === 'number' && inputMeta.shape[2] > 0) {
    inputSize = inputMeta.shape[2];
    offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
    ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
    inputBuffer = new Float32Array(3 * inputSize * inputSize);
  }
  const output = session.outputMetadata[0];
  if (!output?.isTensor || output.shape.length !== 3 ||
      (output.shape[1] !== 5 && output.shape[1] !== 84)) {
    throw new Error('Expected a single-class baseball or 80-class COCO YOLO model');
  }
  // Always preserve aspect ratio with YOLO letterbox padding (114 gray) so vertical phone videos never squash the baseball.
  preserveAspectRatio = true;

  // Signal READY immediately so the user can measure right away without waiting for secondary models.
  workerScope.postMessage({ type: 'READY', backend: activeBackend, actionModelReady: false });

  // Load the lightweight 160x160 pitching action trigger model lazily in the background.
  if (actionModelUrl) {
    void (async () => {
      try {
        const actionBytes = await fetchModelBytes(actionModelUrl);
        if (actionBytes) {
          actionSession = await ort.InferenceSession.create(actionBytes, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'basic',
          });
          actionCanvas = new OffscreenCanvas(ACTION_SIZE, ACTION_SIZE);
          actionCtx = actionCanvas.getContext('2d', { willReadFrequently: true });
          actionBuffer = new Float32Array(3 * ACTION_SIZE * ACTION_SIZE);
        }
      } catch {
        actionSession = null;
      }
    })();
  }
}

function preprocess(bitmap: ImageBitmap) {
  if (!ctx || !inputBuffer) throw new Error('Canvas buffer not initialized');
  // 自動「投球通道變焦裁切 (Tunnel-Zoom)」：對完整影片裁切上中段投球走廊 (x:10%-90%, y:10%-65%)，
  // 讓 320x320 模型獲得等效 ~640x640 的小球像素解析度，同時維持 1.6 GFLOPs (20 FPS) 即時速度並濾除白球鞋
  const useTunnelZoom = inputSize <= 384 && bitmap.height >= 480 && bitmap.width >= 360;
  const srcX = useTunnelZoom ? Math.floor(bitmap.width * 0.10) : 0;
  const srcY = useTunnelZoom ? Math.floor(bitmap.height * 0.10) : 0;
  const srcW = useTunnelZoom ? Math.floor(bitmap.width * 0.80) : bitmap.width;
  const srcH = useTunnelZoom ? Math.floor(bitmap.height * 0.55) : bitmap.height;

  const scale = Math.min(inputSize / srcW, inputSize / srcH);
  const drawWidth = preserveAspectRatio ? Math.max(1, Math.round(srcW * scale)) : inputSize;
  const drawHeight = preserveAspectRatio ? Math.max(1, Math.round(srcH * scale)) : inputSize;
  const padX = Math.floor((inputSize - drawWidth) / 2);
  const padY = Math.floor((inputSize - drawHeight) / 2);
  if (preserveAspectRatio) {
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, inputSize, inputSize);
  }
  ctx.drawImage(bitmap, srcX, srcY, srcW, srcH, padX, padY, drawWidth, drawHeight);
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
    srcX,
    srcY,
    srcW,
    srcH,
    bitmapW: bitmap.width,
    bitmapH: bitmap.height,
  };
}

function parseBestBall(
  output: ort.Tensor,
  origWidth: number,
  origHeight: number,
  timestampMs: number,
  { padX, padY, drawWidth, drawHeight, srcX, srcY, srcW, srcH, bitmapW, bitmapH }: ReturnType<typeof preprocess>
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

  const mapX = origWidth / bitmapW;
  const mapY = origHeight / bitmapH;
  const cropScaleX = srcW / drawWidth;
  const cropScaleY = srcH / drawHeight;
  return {
    x: (srcX + (data[bestIdx] - padX) * cropScaleX) * mapX,
    y: (srcY + (data[anchors + bestIdx] - padY) * cropScaleY) * mapY,
    width: data[2 * anchors + bestIdx] * cropScaleX * mapX,
    height: data[3 * anchors + bestIdx] * cropScaleY * mapY,
    confidence: bestScore,
    timestampMs,
  };
}

async function inferActionPhase(
  bitmap: ImageBitmap
): Promise<{ actionPhase: PitchActionPhase; actionConfidence: number } | null> {
  if (!actionSession || !actionCtx || !actionBuffer) return null;
  try {
    const scale = Math.min(ACTION_SIZE / bitmap.width, ACTION_SIZE / bitmap.height);
    const dw = Math.max(1, Math.round(bitmap.width * scale));
    const dh = Math.max(1, Math.round(bitmap.height * scale));
    const px = Math.floor((ACTION_SIZE - dw) / 2);
    const py = Math.floor((ACTION_SIZE - dh) / 2);
    actionCtx.fillStyle = 'rgb(114,114,114)';
    actionCtx.fillRect(0, 0, ACTION_SIZE, ACTION_SIZE);
    actionCtx.drawImage(bitmap, px, py, dw, dh);
    const { data } = actionCtx.getImageData(0, 0, ACTION_SIZE, ACTION_SIZE);
    const area = ACTION_SIZE * ACTION_SIZE;
    const inv255 = 1 / 255;
    for (let i = 0; i < area; i++) {
      const idx = i * 4;
      actionBuffer[i] = data[idx] * inv255;
      actionBuffer[i + area] = data[idx + 1] * inv255;
      actionBuffer[i + 2 * area] = data[idx + 2] * inv255;
    }
    const tensor = new ort.Tensor('float32', actionBuffer, [1, 3, ACTION_SIZE, ACTION_SIZE]);
    const out = await actionSession.run({ [actionSession.inputNames[0]]: tensor });
    const probs = out[actionSession.outputNames[0]].data as Float32Array;
    let bestIdx = 0;
    let bestProb = probs[0] ?? 0;
    for (let i = 1; i < Math.min(ACTION_LABELS.length, probs.length); i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIdx = i;
      }
    }
    return { actionPhase: ACTION_LABELS[bestIdx], actionConfidence: bestProb };
  } catch {
    return null;
  }
}

workerScope.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  try {
    if (msg.type === 'INIT') {
      await initModel(
        msg.modelUrl,
        msg.wasmPaths,
        msg.inputSize,
        msg.confThreshold,
        msg.actionModelUrl,
      );
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
        const action = msg.detectAction ? await inferActionPhase(msg.bitmap) : null;
        workerScope.postMessage({
          type: 'RESULT',
          detection,
          inferTimeMs: Number((performance.now() - start).toFixed(1)),
          actionPhase: action?.actionPhase,
          actionConfidence: action?.actionConfidence,
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
