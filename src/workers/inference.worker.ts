import * as ort from 'onnxruntime-web';
import { BallDetection, WorkerInMessage, WorkerOutMessage } from '../types';

const workerScope = self as unknown as {
  postMessage: (msg: WorkerOutMessage) => void;
  onmessage: ((ev: MessageEvent<WorkerInMessage>) => void) | null;
};

let session: ort.InferenceSession | null = null;
let inputSize = 320;
let confThreshold = 0.22;
let offscreenCanvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let inputBuffer: Float32Array | null = null;
let prevLumaBuffer: Uint8Array | null = null;

async function initModel(
  modelUrl: string,
  wasmPaths: string,
  size = 320,
  threshold = 0.22
) {
  inputSize = size;
  confThreshold = Math.min(threshold, 0.25);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = wasmPaths;

  offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
  ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  inputBuffer = new Float32Array(3 * inputSize * inputSize);
  prevLumaBuffer = new Uint8Array(inputSize * inputSize);

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

function preprocessAndExtractLuma(bitmap: ImageBitmap): {
  tensor: ort.Tensor;
  currLuma: Uint8Array;
} {
  if (!ctx || !inputBuffer) throw new Error('Canvas buffer not initialized');
  ctx.drawImage(bitmap, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
  const area = inputSize * inputSize;
  const inv255 = 1 / 255;
  const currLuma = new Uint8Array(area);

  for (let i = 0; i < area; i++) {
    const px = i * 4;
    const r = data[px];
    const g = data[px + 1];
    const b = data[px + 2];
    inputBuffer[i] = r * inv255;
    inputBuffer[i + area] = g * inv255;
    inputBuffer[i + 2 * area] = b * inv255;
    currLuma[i] = (r * 77 + g * 150 + b * 29) >> 8;
  }
  return {
    tensor: new ort.Tensor('float32', inputBuffer, [1, 3, inputSize, inputSize]),
    currLuma,
  };
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

/**
 * 動態模糊棒球輔助偵測：當高速飛行造成單幀模糊使 YOLO 分數略降時，
 * 檢查局部高對比明亮球體移動區塊（排除鏡頭晃動或人體大面積移動）
 */
function detectMotionBlurredBallFallback(
  currLuma: Uint8Array,
  prevLuma: Uint8Array | null,
  origWidth: number,
  origHeight: number,
  timestampMs: number
): BallDetection | null {
  if (!prevLuma) return null;
  const size = inputSize;
  const area = size * size;

  let activePixels = 0;
  let sumX = 0;
  let sumY = 0;
  let sumWeight = 0;
  let minX = size;
  let maxX = 0;
  let minY = size;
  let maxY = 0;

  // 僅掃描畫面中央 84% 投球通道區域
  const margin = Math.floor(size * 0.08);
  for (let y = margin; y < size - margin; y += 2) {
    const rowOffset = y * size;
    for (let x = margin; x < size - margin; x += 2) {
      const idx = rowOffset + x;
      const luma = currLuma[idx];
      const diff = luma - prevLuma[idx];
      // 棒球為明亮物體 (luma >= 145) 且與前一幀有顯著正向亮度差 (diff >= 38)
      if (luma >= 145 && diff >= 38) {
        activePixels += 1;
        const w = diff;
        sumX += x * w;
        sumY += y * w;
        sumWeight += w;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // 若移動像素過少(雜訊)或過多(鏡頭晃動/人體移動)，立即拒絕
  if (activePixels < 4 || activePixels > area * 0.008 || sumWeight === 0) {
    return null;
  }

  const blobW = maxX - minX + 4;
  const blobH = maxY - minY + 4;
  // 球體或動態模糊球跡在 320x320 下尺寸應介於 4px ~ 52px 之間
  if (blobW > 52 || blobH > 52 || blobW < 4 || blobH < 4) {
    return null;
  }

  const scaleX = origWidth / size;
  const scaleY = origHeight / size;
  const cx = (sumX / sumWeight) * scaleX;
  const cy = (sumY / sumWeight) * scaleY;

  return {
    x: cx,
    y: cy,
    width: Math.max(10, blobW * scaleX),
    height: Math.max(10, blobH * scaleY),
    confidence: 0.42,
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
        const { tensor, currLuma } = preprocessAndExtractLuma(msg.bitmap);
        const results = await session.run({ [session.inputNames[0]]: tensor });
        let detection = parseBestBall(
          results[session.outputNames[0]],
          msg.origWidth,
          msg.origHeight,
          msg.timestampMs
        );
        if (!detection && prevLumaBuffer) {
          detection = detectMotionBlurredBallFallback(
            currLuma,
            prevLumaBuffer,
            msg.origWidth,
            msg.origHeight,
            msg.timestampMs
          );
        }
        prevLumaBuffer = currLuma;
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
