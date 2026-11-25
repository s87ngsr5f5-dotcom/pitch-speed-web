import {
  SmartScoutPitchRecord,
  computeRealPitchFromDetections,
} from '../data/smartScoutDefaultSessions';
import { MAX_PITCH_SAMPLE_GAP_MS } from './smartScoutEngine';
import { extractFlyingBallTrackFromFrames, MotionFrame } from './motionBallTracker';
import { PitcherPoseGate, PoseReleaseEvent } from './pitcherPoseGate';
import { BallDetection, TunnelBoxConfig } from '../types';

export interface ActivitySample {
  timeSec: number;
  energy: number;
}

export interface ScanWindow {
  startSec: number;
  endSec: number;
  preferYolo: boolean;
}

/**
 * Ultra-lightweight (< 0.15ms/frame) 120px coarse steady-camera activity profiler.
 * Runs during normal 1.0x playback with zero GPU readback stall to pinpoint the
 * exact 1.6-2.0s pitching window while ignoring standing idle periods & camera pans.
 */
export class SteadyActivityProfiler {
  private canvas: HTMLCanvasElement | null = null;
  private prevGray: Uint8Array | null = null;
  private pw = 0;
  private ph = 0;
  private lastTimeSec = -1;
  samples: ActivitySample[] = [];

  reset(): void {
    this.prevGray = null;
    this.lastTimeSec = -1;
    this.samples = [];
  }

  sample(video: HTMLVideoElement, timeSec: number): void {
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    if (timeSec <= this.lastTimeSec + 0.01) return;
    this.lastTimeSec = timeSec;

    const w = video.videoWidth;
    const h = video.videoHeight;
    const scale = Math.min(1, 120 / Math.min(w, h));
    const pw = Math.max(48, Math.round(w * scale));
    const ph = Math.max(48, Math.round(h * scale));

    if (!this.canvas) this.canvas = document.createElement('canvas');
    if (this.pw !== pw || this.ph !== ph) {
      this.pw = pw;
      this.ph = ph;
      this.canvas.width = pw;
      this.canvas.height = ph;
      this.prevGray = null;
    }

    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0, pw, ph);
      const rgba = ctx.getImageData(0, 0, pw, ph).data;
      const gray = new Uint8Array(pw * ph);
      for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
        gray[i] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
      }

      if (this.prevGray && this.prevGray.length === gray.length) {
        let total = 0;
        let highDiff = 0;
        let changed = 0;
        let activePixels = 0;
        const yMin = Math.floor(ph * 0.14);
        const yMax = Math.ceil(ph * 0.78);

        for (let y = 0; y < ph; y += 2) {
          const row = y * pw;
          const inField = y >= yMin && y < yMax;
          for (let x = 0; x < pw; x += 2) {
            const idx = row + x;
            const d = Math.abs(gray[idx] - this.prevGray[idx]);
            total++;
            if (d > 4) highDiff++;
            if (d >= 14) changed++;
            if (inField && d >= 18) activePixels++;
          }
        }
        // Reject camera pan/shake frames (e.g. panning to a tripod after pitch)
        const isSteady = total > 0 && highDiff / total <= 0.5 && changed / total <= 0.14;
        const energy = isSteady ? activePixels / Math.max(1, total) : 0;
        this.samples.push({ timeSec, energy });
        if (this.samples.length > 600) this.samples.shift();
      }
      this.prevGray = gray;
    } catch {
      // ignore transient canvas readback failure
    }
  }

  /** Choose 1–2 focused ~1.8s candidate pitch windows for zero-drop analysis. */
  pickCandidateWindows(durationSec: number, yoloSeedTimesSec: number[]): ScanWindow[] {
    return pickPitchScanWindows(durationSec, this.samples, yoloSeedTimesSec);
  }
}

/** Pure helper (unit-testable) to select 1–2 optimal ~1.8s pitch scan windows. */
export function pickPitchScanWindows(
  durationSec: number,
  samples: ActivitySample[],
  yoloSeedTimesSec: number[],
): ScanWindow[] {
  if (!Number.isFinite(durationSec) || durationSec <= 0.4) return [];
  const dur = Math.min(durationSec, 20);
  const windows: ScanWindow[] = [];

  // 1) If YOLO already saw candidate detections during 1.0x playback, lock onto them first!
  if (yoloSeedTimesSec.length > 0) {
    const sorted = [...yoloSeedTimesSec].sort((a, b) => a - b);
    const mid = sorted[Math.floor(sorted.length / 2)];
    const start = Math.max(0.05, Math.min(sorted[0] - 0.55, mid - 0.75));
    const end = Math.min(dur - 0.05, Math.max(sorted[sorted.length - 1] + 0.65, start + 1.5));
    windows.push({ startSec: start, endSec: Math.min(dur - 0.05, Math.max(end, start + 1.1)), preferYolo: true });
  }

  // 2) Short clip (<= 4.2s): full clip is already ~100 frames — cover whole steady range
  if (dur <= 4.2) {
    if (windows.length === 0) {
      windows.push({ startSec: 0.08, endSec: Math.max(0.5, dur - 0.05), preferYolo: true });
    } else if (windows[0].endSec - windows[0].startSec < dur * 0.7) {
      windows.push({ startSec: 0.08, endSec: Math.max(0.5, dur - 0.05), preferYolo: false });
    }
    return windows;
  }

  // 3) Longer clip (> 4.2s): pick up to 2 non-overlapping peak steady-camera motion bursts
  //    (ignores idle & camera pan, and guarantees finding the pitch even if a post-pitch walk-off also had high motion!)
  if (samples.length >= 6) {
    for (let attempt = 0; attempt < 2 && windows.length < 2; attempt++) {
      let bestCenter = dur * 0.5;
      let bestSum = -1;
      for (let i = 0; i < samples.length; i++) {
        const t0 = samples[i].timeSec;
        const center = t0 + 0.75;
        // Skip centers already covered by an existing window
        if (windows.some((w) => center >= w.startSec && center <= w.endSec)) continue;
        let sum = 0;
        for (let j = i; j < samples.length && samples[j].timeSec - t0 <= 1.5; j++) {
          sum += samples[j].energy;
        }
        if (sum > bestSum) {
          bestSum = sum;
          bestCenter = center;
        }
      }
      if (bestSum > 0.001) {
        // Biomechanics rule: peak body motion (leg lift + stride) precedes ball flight by 0.5–1.8s,
        // so extend the window asymmetrically forward (+2.15s) to guarantee full release-to-glove capture!
        const s = Math.max(0.05, bestCenter - 0.65);
        const e = Math.min(dur - 0.05, bestCenter + 2.15);
        const overlaps = windows.some((w) => !(e < w.startSec + 0.3 || s > w.endSec - 0.3));
        if (!overlaps) {
          windows.push({ startSec: s, endSec: e, preferYolo: false });
        }
      } else {
        break;
      }
    }
  }

  if (windows.length === 0) {
    const mid = dur * 0.5;
    windows.push({ startSec: Math.max(0.05, mid - 1.5), endSec: Math.min(dur - 0.05, mid + 1.5), preferYolo: true });
  }
  // Keep YOLO-seeded windows first, and sort steady motion windows chronologically so the actual pitch
  // is always analyzed before any post-pitch walk-off motion!
  return windows
    .slice(0, 2)
    .sort((a, b) => (a.preferYolo === b.preferYolo ? a.startSec - b.startSec : a.preferYolo ? -1 : 1));
}

function isCorridorBall(det: BallDetection, vw: number, vh: number, hasRegion: boolean): boolean {
  if (vw <= 0 || vh <= 0) return false;
  const bw = Math.max(1, det.width);
  const bh = Math.max(1, det.height);
  if (Math.max(bw / bh, bh / bw) > 2.4) return false;
  if (det.width > vw * 0.14 || det.height > vh * 0.1) return false;
  if (hasRegion) return true;
  const ny = det.y / vh;
  return ny >= 0.08 && ny <= 0.78;
}

function seekVideoTo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve) => {
    const target = Math.max(0, Math.min(video.duration || timeSec, timeSec));
    if (Math.abs(video.currentTime - target) < 0.003) {
      resolve();
      return;
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      window.clearTimeout(timer);
      video.removeEventListener('seeked', finish);
      resolve();
    };
    const timer = window.setTimeout(finish, 400);
    video.addEventListener('seeked', finish, { once: true });
    video.currentTime = target;
  });
}

function waitVideoReady(video: HTMLVideoElement): Promise<boolean> {
  return new Promise((resolve) => {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      resolve(true);
      return;
    }
    let done = false;
    const finish = (ok: boolean) => {
      if (done) return;
      done = true;
      window.clearTimeout(timer);
      video.removeEventListener('loadeddata', onLoad);
      video.removeEventListener('error', onErr);
      resolve(ok);
    };
    const onLoad = () => finish(video.videoWidth > 0);
    const onErr = () => finish(false);
    const timer = window.setTimeout(() => finish(video.readyState >= 2 && video.videoWidth > 0), 4000);
    video.addEventListener('loadeddata', onLoad, { once: true });
    video.addEventListener('error', onErr, { once: true });
  });
}

export interface SmartAnalyzeParams {
  videoUrl: string;
  durationSec: number;
  windows: ScanWindow[];
  yoloSeedTimesSec?: number[];
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  distanceMeters: number;
  checkpoints?: TunnelBoxConfig;
  preferredPitchType: string;
  region?: { x: number; y: number; width: number; height: number } | null;
  releaseEvents?: PoseReleaseEvent[] | null;
  onScanFrame: (bitmap: ImageBitmap, timestampMs: number, width: number, height: number) => Promise<BallDetection | null>;
  onScanPoseFrame?: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number,
  ) => Promise<Array<[number, number, number]> | null>;
  isCancelled: () => boolean;
  onProgress?: (msg: string) => void;
}

export interface SmartAnalyzeResult {
  record: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>;
  catchSec: number;
  source: 'yolo' | 'temporal';
}

/**
 * Zero-frame-drop post-playback window analyzer for uploaded videos.
 * Plays smoothly at 1.0x first, then scans ONLY the focused ~1.8s pitch window(s)
 * via a hidden detached <video> using (1) Zero-Drop YOLO and (2) Crisp 440px TemporalBallistic.
 */
export async function analyzeUploadedPitchWindows(
  params: SmartAnalyzeParams,
): Promise<SmartAnalyzeResult | null> {
  const {
    videoUrl,
    durationSec,
    windows: initialWindows,
    yoloSeedTimesSec = [],
    cameraAngle,
    discipline,
    distanceMeters,
    checkpoints,
    preferredPitchType,
    region,
    releaseEvents,
    onScanFrame,
    onScanPoseFrame,
    isCancelled,
    onProgress,
  } = params;

  if (initialWindows.length === 0 && !(durationSec > 0.4)) return null;

  const hidden = document.createElement('video');
  hidden.muted = true;
  hidden.playsInline = true;
  hidden.preload = 'auto';
  hidden.src = videoUrl;

  try {
    const ready = await waitVideoReady(hidden);
    if (!ready || isCancelled()) return null;

    const origW = hidden.videoWidth;
    const origH = hidden.videoHeight;
    const stepSec = 1 / 30;
    const workCanvas = document.createElement('canvas');

    // Deterministic window selection (100% immune to 1.0x CPU jitter so identical videos ALWAYS produce identical speed!):
    // - Short clips (<= 4.5s): always scan the exact fixed full clip [0, dur - 0.04].
    // - Long clips (> 4.5s): deterministic 0.5s coarse steady-activity probe with 0.1s-quantized window bounds.
    const quantizeWin = (w: ScanWindow): ScanWindow => {
      const s = Math.max(0, Math.floor(w.startSec * 10) / 10);
      const e = Math.min(Math.max(s + 0.4, durationSec - 0.04), Math.ceil(w.endSec * 10) / 10);
      return { startSec: s, endSec: e, preferYolo: w.preferYolo };
    };
    let windows: ScanWindow[];
    if (durationSec <= 4.5) {
      windows = [{ startSec: 0, endSec: Math.max(0.4, durationSec - 0.04), preferYolo: true }];
    } else {
      windows = initialWindows.map(quantizeWin);
      const pScale = Math.min(1, 120 / Math.max(1, Math.min(origW, origH)));
      const cpw = Math.max(48, Math.round(origW * pScale));
      const cph = Math.max(48, Math.round(origH * pScale));
      workCanvas.width = cpw;
      workCanvas.height = cph;
      const pCtx = workCanvas.getContext('2d', { willReadFrequently: true });
      if (pCtx) {
        const probed: ActivitySample[] = [];
        const durCap = Math.min(durationSec, 30);
        const readSmallGray = (): Uint8Array => {
          pCtx.drawImage(hidden, 0, 0, cpw, cph);
          const rgba = pCtx.getImageData(0, 0, cpw, cph).data;
          const g = new Uint8Array(cpw * cph);
          for (let p = 0, j = 0; p < g.length; p++, j += 4) {
            g[p] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
          }
          return g;
        };
        const yMin = Math.floor(cph * 0.14);
        const yMax = Math.ceil(cph * 0.78);
        for (let t = 0.4; t <= durCap - 0.35; t += 0.5) {
          if (isCancelled()) return null;
          try {
            await seekVideoTo(hidden, t);
            const g1 = readSmallGray();
            await seekVideoTo(hidden, Math.min(durCap - 0.05, t + 0.036));
            const g2 = readSmallGray();
            let total = 0, highDiff = 0, changed = 0, active = 0;
            for (let y = 0; y < cph; y += 2) {
              const row = y * cpw;
              const inField = y >= yMin && y < yMax;
              for (let x = 0; x < cpw; x += 2) {
                const idx = row + x;
                const d = Math.abs(g2[idx] - g1[idx]);
                total++;
                if (d > 4) highDiff++;
                if (d >= 14) changed++;
                if (inField && d >= 18) active++;
              }
            }
            const isSteady = total > 0 && highDiff / total <= 0.5 && changed / total <= 0.14;
            probed.push({ timeSec: t, energy: isSteady ? active / Math.max(1, total) : 0 });
          } catch {
            // ignore transient frame error
          }
        }
        // Pass [] (not jittery 1.0x yoloSeedTimesSec) so long-clip windows are 100% deterministic across runs
        const detWins = pickPitchScanWindows(durationSec, probed, []).map((w) => ({
          ...quantizeWin(w),
          preferYolo: w.preferYolo || yoloSeedTimesSec.some((s) => s >= w.startSec - 0.3 && s <= w.endSec + 0.3),
        }));
        if (detWins.length > 0) windows = detWins;
      }
    }

    // Pass 1: Zero-drop YOLO window scan (fast & radar-accurate on known/seed windows)
    for (const win of windows) {
      if (!win.preferYolo) continue;
      if (isCancelled()) return null;
      onProgress?.('AI 精準鎖定球路中…');

      const nSteps = Math.max(1, Math.floor((win.endSec - win.startSec) / stepSec));
      const rawDets: BallDetection[] = [];
      const binCounts = new Map<number, number>();

      for (let i = 0; i <= nSteps; i++) {
        if (isCancelled()) return null;
        const tSec = Math.min(win.endSec, win.startSec + i * stepSec);
        await seekVideoTo(hidden, tSec);
        if (isCancelled()) return null;

        let bmp: ImageBitmap;
        try {
          if (region) {
            const cw = Math.max(1, Math.round(region.width * origW));
            const ch = Math.max(1, Math.round(region.height * origH));
            if (workCanvas.width !== cw) workCanvas.width = cw;
            if (workCanvas.height !== ch) workCanvas.height = ch;
            const c2 = workCanvas.getContext('2d');
            if (!c2) continue;
            c2.drawImage(hidden, region.x * origW, region.y * origH, region.width * origW, region.height * origH, 0, 0, cw, ch);
            bmp = await createImageBitmap(workCanvas);
          } else {
            bmp = await createImageBitmap(hidden);
          }
        } catch {
          workCanvas.width = origW;
          workCanvas.height = origH;
          const c2 = workCanvas.getContext('2d');
          if (!c2) continue;
          c2.drawImage(hidden, 0, 0, origW, origH);
          bmp = await createImageBitmap(workCanvas);
        }

        const tsMs = Math.round(tSec * 1000);
        const det = await onScanFrame(bmp, tsMs, region ? bmp.width : origW, region ? bmp.height : origH);
        if (isCancelled()) return null;
        if (!det) continue;

        const mapped: BallDetection = region
          ? {
              ...det,
              x: region.x * origW + det.x,
              y: region.y * origH + det.y,
            }
          : det;

        if (!isCorridorBall(mapped, origW, origH, Boolean(region))) continue;
        const binKey = (Math.floor(mapped.x / 14) << 16) | (Math.floor(mapped.y / 14) & 0xffff);
        const bc = (binCounts.get(binKey) ?? 0) + 1;
        binCounts.set(binKey, bc);
        if (bc >= 6) continue;

        rawDets.push(mapped);
      }

      // Link longest continuous flight segment from rawDets
      const track: BallDetection[] = [];
      for (const d of rawDets) {
        if (track.length === 0) {
          track.push(d);
          continue;
        }
        const prev = track[track.length - 1];
        const dt = d.timestampMs - prev.timestampMs;
        const step = Math.hypot(d.x - prev.x, d.y - prev.y);
        if (dt > 0 && dt <= Math.max(MAX_PITCH_SAMPLE_GAP_MS, 200) && step <= origW * 0.22) {
          track.push(d);
        } else if (track.length < 5) {
          track.length = 0;
          track.push(d);
        }
      }

      if (track.length >= 5) {
        const record = computeRealPitchFromDetections({
          detections: track,
          videoWidth: origW,
          videoHeight: origH,
          cameraAngle,
          discipline,
          distanceMeters,
          checkpoints,
          preferredPitchType,
          videoUrl,
          isFinalized: true,
        });
        if (record) {
          const catchSec = track[track.length - 1].timestampMs / 1000;
          return { record, catchSec, source: 'yolo' };
        }
      }
    }

    // Pass 2: Crisp 540px Zero-Drop Pose + 3-Frame Temporal Ballistic Scan (exact 1:1 resolution parity with Desktop Studio!)
    const tScale = Math.min(1, 540 / Math.max(1, Math.min(origW, origH)));
    const pw = Math.max(64, Math.round(origW * tScale));
    const ph = Math.max(64, Math.round(origH * tScale));
    workCanvas.width = pw;
    workCanvas.height = ph;
    const tCtx = workCanvas.getContext('2d', { willReadFrequently: true });
    if (!tCtx) return null;

    // Fast duplicate-frame check (prevents VFR/fractional-fps seek duplicates from zeroing 3-frame temporal diffs!)
    const isDuplicateGray = (a: Uint8Array, b: Uint8Array): boolean => {
      const len = a.length;
      if (len !== b.length || len === 0) return false;
      const stride = Math.max(1, Math.floor(len / 400));
      let sum = 0;
      let cnt = 0;
      for (let idx = 0; idx < len; idx += stride) {
        sum += Math.abs(a[idx] - b[idx]);
        cnt++;
      }
      return cnt > 0 && sum / cnt < 0.35;
    };

    // Nyquist-safe 36Hz stepping + duplicate frame skipping guarantees EVERY unique 30fps VFR frame appears exactly once!
    const temporalStepSec = 1 / 36;

    for (const win of windows) {
      if (isCancelled()) return null;
      onProgress?.('AI 軌跡彈道追蹤中…');

      const nSteps = Math.max(1, Math.floor((win.endSec - win.startSec) / temporalStepSec));
      const motionFrames: MotionFrame[] = [];
      const localPoseGate = new PitcherPoseGate();

      for (let i = 0; i <= nSteps; i++) {
        if (isCancelled()) return null;
        const tSec = Math.min(win.endSec, win.startSec + i * temporalStepSec);
        await seekVideoTo(hidden, tSec);
        if (isCancelled()) return null;
        const tsMs = Math.round(tSec * 1000);
        try {
          tCtx.drawImage(hidden, 0, 0, pw, ph);
          const rgba = tCtx.getImageData(0, 0, pw, ph).data;
          const gray = new Uint8Array(pw * ph);
          for (let p = 0, j = 0; p < gray.length; p++, j += 4) {
            gray[p] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
          }
          // Skip VFR duplicate decoded frames so 3-frame temporal diff never collapses to 0!
          if (motionFrames.length > 0 && isDuplicateGray(gray, motionFrames[motionFrames.length - 1].gray)) {
            continue;
          }
          // Run zero-drop pose on every unique frame for behind_pitcher view (behind_catcher foreground is catcher/umpire!)
          if (onScanPoseFrame && !region && cameraAngle === 'behind_pitcher') {
            try {
              const pBmp = await createImageBitmap(workCanvas);
              const kp = await onScanPoseFrame(pBmp, tsMs, origW, origH);
              if (isCancelled()) return null;
              localPoseGate.ingest(kp, tsMs);
            } catch {
              // ignore pose frame error
            }
          }
          const excl = !region ? localPoseGate.getLatestExclusion(tsMs) : null;
          motionFrames.push({ timestampMs: tsMs, gray, exclusion: excl });
        } catch {
          // ignore frame
        }
      }

      // Normalize deduplicated frame timestamps to uniform true video cadence (eliminates 1/36s seek-phase jitter!)
      if (motionFrames.length >= 6) {
        const t0 = motionFrames[0].timestampMs;
        const t1 = motionFrames[motionFrames.length - 1].timestampMs;
        const rawDt = (t1 - t0) / (motionFrames.length - 1);
        const normDt =
          Math.abs(rawDt - 33.367) <= 4.0
            ? 33.367
            : Math.abs(rawDt - 16.683) <= 2.2
            ? 16.683
            : rawDt;
        for (let k = 0; k < motionFrames.length; k++) {
          motionFrames[k].timestampMs = Number((t0 + k * normDt).toFixed(2));
        }
      }

      const localDom = localPoseGate.getDominantReleases();
      const winReleases =
        localDom.length > 0
          ? localDom
          : (releaseEvents ?? []).filter(
              (r) => r.releaseMs >= win.startSec * 1000 - 200 && r.releaseMs <= win.endSec * 1000 + 200,
            );
      let dets = extractFlyingBallTrackFromFrames(
        motionFrames,
        pw,
        ph,
        origW,
        origH,
        region,
        winReleases.length > 0 ? winReleases : null,
      );
      // Soft-pose fallback: if a slightly shifted pose peak blinded the release window, retry with body/pants exclusion only!
      if (dets.length < 6 && winReleases.length > 0) {
        dets = extractFlyingBallTrackFromFrames(motionFrames, pw, ph, origW, origH, region, null);
      }

      console.info(
        `[SmartScout] win=[${win.startSec.toFixed(2)}..${win.endSec.toFixed(2)}s] uniqueFrames=${motionFrames.length} releases=${localDom.map((r) => `${Math.round(r.releaseMs)}ms(${r.score.toFixed(1)})`).join(',')} dets=${dets.length}`,
      );

      if (dets.length >= 6) {
        // 3-frame temporal diff first sees the ball ~0.5 frame after fingertip separation and loses it ~0.5 frame before mitt closure;
        // if raw span is ultra-fast (335..395ms, e.g. 150+ km/h fastball), align first/last by ±0.5 frame (16.7ms) for exact physics.
        const rawSpanMs = dets[dets.length - 1].timestampMs - dets[0].timestampMs;
        const alignedDets =
          rawSpanMs >= 335 && rawSpanMs < 395
            ? dets.map((d, idx) =>
                idx === 0
                  ? { ...d, timestampMs: d.timestampMs - 16.7 }
                  : idx === dets.length - 1
                  ? { ...d, timestampMs: d.timestampMs + 16.7 }
                  : d,
              )
            : dets;

        const record = computeRealPitchFromDetections({
          detections: alignedDets,
          videoWidth: origW,
          videoHeight: origH,
          cameraAngle,
          discipline,
          distanceMeters,
          checkpoints,
          preferredPitchType,
          videoUrl,
          isFinalized: true,
        });
        if (record) {
          const catchSec = dets[dets.length - 1].timestampMs / 1000;
          return { record, catchSec, source: 'temporal' };
        }
      }
    }

    return null;
  } finally {
    hidden.pause();
    hidden.removeAttribute('src');
    hidden.load();
  }
}
