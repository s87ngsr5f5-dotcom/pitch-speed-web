import * as ort from 'onnxruntime-web/wasm';
import { parsePitcherPose } from '../core/pitcherPoseGate';

type PoseInMsg =
  | { type: 'INIT'; poseModelUrl: string; wasmPaths: string }
  | { type: 'POSE'; bitmap: ImageBitmap; timestampMs: number; origWidth: number; origHeight: number };

type PoseOutMsg =
  | { type: 'READY' }
  | { type: 'POSE_RESULT'; timestampMs: number; pose: Array<[number, number, number]> | null };

const workerScope = self as unknown as {
  postMessage: (msg: PoseOutMsg) => void;
  onmessage: ((ev: MessageEvent<PoseInMsg>) => void) | null;
};

const POSE_SIZE = 320;
let poseSession: ort.InferenceSession | null = null;
let poseCanvas: OffscreenCanvas | null = null;
let poseCtx: OffscreenCanvasRenderingContext2D | null = null;
let poseBuffer: Float32Array | null = null;

async function fetchBytes(url: string): Promise<Uint8Array | null> {
  try {
    if ('caches' in self) {
      const cached = await caches.match(url);
      if (cached && cached.ok) return new Uint8Array(await cached.arrayBuffer());
    }
    const resp = await fetch(url);
    if (resp.ok) return new Uint8Array(await resp.arrayBuffer());
  } catch {
    // ignore
  }
  return null;
}

workerScope.onmessage = async (event: MessageEvent<PoseInMsg>) => {
  const msg = event.data;
  if (msg.type === 'INIT') {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = msg.wasmPaths;
      const bytes = await fetchBytes(msg.poseModelUrl);
      if (!bytes) return;
      poseSession = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
      });
      poseCanvas = new OffscreenCanvas(POSE_SIZE, POSE_SIZE);
      poseCtx = poseCanvas.getContext('2d', { willReadFrequently: true });
      poseBuffer = new Float32Array(3 * POSE_SIZE * POSE_SIZE);
      workerScope.postMessage({ type: 'READY' });
    } catch {
      poseSession = null;
    }
    return;
  }

  if (msg.type === 'POSE') {
    try {
      if (!poseSession || !poseCtx || !poseBuffer) {
        workerScope.postMessage({ type: 'POSE_RESULT', timestampMs: msg.timestampMs, pose: null });
        return;
      }
      const scale = Math.min(POSE_SIZE / msg.bitmap.width, POSE_SIZE / msg.bitmap.height);
      const dw = Math.max(1, Math.round(msg.bitmap.width * scale));
      const dh = Math.max(1, Math.round(msg.bitmap.height * scale));
      const px = Math.floor((POSE_SIZE - dw) / 2);
      const py = Math.floor((POSE_SIZE - dh) / 2);
      poseCtx.fillStyle = 'rgb(114,114,114)';
      poseCtx.fillRect(0, 0, POSE_SIZE, POSE_SIZE);
      poseCtx.drawImage(msg.bitmap, px, py, dw, dh);
      const { data } = poseCtx.getImageData(0, 0, POSE_SIZE, POSE_SIZE);
      const area = POSE_SIZE * POSE_SIZE;
      const inv255 = 1 / 255;
      for (let i = 0; i < area; i++) {
        const idx = i * 4;
        poseBuffer[i] = data[idx] * inv255;
        poseBuffer[i + area] = data[idx + 1] * inv255;
        poseBuffer[i + 2 * area] = data[idx + 2] * inv255;
      }
      const tensor = new ort.Tensor('float32', poseBuffer, [1, 3, POSE_SIZE, POSE_SIZE]);
      const out = await poseSession.run({ [poseSession.inputNames[0]]: tensor });
      const o = out[poseSession.outputNames[0]];
      const bitmapToOrig = msg.origWidth / msg.bitmap.width;
      const kp = parsePitcherPose(o.data as Float32Array, o.dims[2], {
        padX: px,
        padY: py,
        scale: scale / bitmapToOrig,
        origW: msg.origWidth,
        origH: msg.origHeight,
      });
      workerScope.postMessage({
        type: 'POSE_RESULT',
        timestampMs: msg.timestampMs,
        pose: kp ? kp.map((p) => [p[0], p[1], p[2]] as [number, number, number]) : null,
      });
    } catch {
      workerScope.postMessage({ type: 'POSE_RESULT', timestampMs: msg.timestampMs, pose: null });
    } finally {
      msg.bitmap.close();
    }
  }
};
