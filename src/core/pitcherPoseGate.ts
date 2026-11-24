/**
 * 投手動作出手閘門（Web 版，與桌面 pitcher_pose_gate.py、iOS PitcherPoseGate.swift 同邏輯）
 *
 * 四階段：IDLE → WINDUP（抬腿 / 拉臂）→ RELEASE（投球手腕甩動峰值）→ FOLLOW_THROUGH
 * 規則：
 *  1. 出手 = 投球手腕速度（軀幹高度 / 秒）× 手腕高度加權 的局部峰值，且 ≥ 門檻。
 *  2. 出手前 1.2 秒內必須看到準備動作（抬腿或手臂移動）至少 3 個樣本，否則視為假出手。
 *  3. 兩次出手至少間隔 1.5 秒。
 *  4. 測速只接受 [出手 − 70ms, 出手 + 1150ms] 內的球點。
 *  5. 最近 2 秒完全看不到投手時，閘門放行（不擋測速）。
 */

// COCO 17 點索引
export const KP = {
  L_SHOULDER: 5,
  R_SHOULDER: 6,
  L_WRIST: 9,
  R_WRIST: 10,
  L_HIP: 11,
  R_HIP: 12,
  L_KNEE: 13,
  R_KNEE: 14,
} as const;

/** 17 個關鍵點，每點 [x, y, conf]（像素座標，y 向下） */
export type PoseKeypoints = ReadonlyArray<readonly [number, number, number]>;

export interface PoseReleaseEvent {
  releaseMs: number;
  throwingHand: 'R' | 'L';
  score: number;
}

interface Sample {
  ts: number;
  personSeen: boolean;
  rSpeed: number;
  lSpeed: number;
  rAbove: number;
  lAbove: number;
  kneeLift: number;
}

const MIN_CONF = 0.25;

export class PitcherPoseGate {
  static readonly WINDOW_BEFORE_MS = 70;
  static readonly WINDOW_AFTER_MS = 1150;
  static readonly RELEASE_SCORE_THRESHOLD = 2.0;
  static readonly WINDUP_LOOKBACK_MS = 1200;
  static readonly MIN_WINDUP_SAMPLES = 3;
  static readonly MIN_RELEASE_GAP_MS = 1500;
  static readonly POSE_AVAILABLE_MS = 2000;

  private samples: Sample[] = [];
  private lastKp: PoseKeypoints | null = null;
  private lastKpTs = -1;
  private lastPersonSeenTs = -1_000_000;
  releases: PoseReleaseEvent[] = [];

  reset(): void {
    this.samples = [];
    this.releases = [];
    this.lastKp = null;
    this.lastKpTs = -1;
    this.lastPersonSeenTs = -1_000_000;
  }

  isPoseAvailable(nearMs: number): boolean {
    return Math.abs(nearMs - this.lastPersonSeenTs) <= PitcherPoseGate.POSE_AVAILABLE_MS;
  }

  isMeasurementAllowed(ts: number): boolean {
    if (!this.isPoseAvailable(ts) || this.releases.length === 0) return true;
    return this.releases.some(
      (r) => ts >= r.releaseMs - PitcherPoseGate.WINDOW_BEFORE_MS && ts <= r.releaseMs + PitcherPoseGate.WINDOW_AFTER_MS,
    );
  }

  /**
   * 依出手事件裁切「出手前」準備期雜點：
   * - 若有明確出手事件落在軌跡開頭附近或中段（[firstTs - 150ms, lastTs - 180ms]），
   *   裁掉 releaseMs - 70ms 之前還握在手上的幀，並完整保留出手後的飛行軌跡（絕不切斷進壘尾段）。
   * - 若看不到投手或該球未捕捉到對應出手峰值，交由物理軌跡引擎把關，絕不把真實飛行軌跡清空。
   */
  filter<T extends { timestampMs: number }>(detections: T[]): T[] {
    if (detections.length === 0 || this.releases.length === 0) return detections;
    const firstTs = detections[0].timestampMs;
    const lastTs = detections[detections.length - 1].timestampMs;
    if (!this.isPoseAvailable(firstTs) && !this.isPoseAvailable(lastTs)) {
      return detections;
    }
    const matching = this.releases.filter(
      (r) => r.releaseMs >= firstTs - 150 && r.releaseMs <= lastTs - 180,
    );
    if (matching.length === 0) return detections;
    const rel = matching[matching.length - 1];
    const kept = detections.filter(
      (d) => d.timestampMs >= rel.releaseMs - PitcherPoseGate.WINDOW_BEFORE_MS,
    );
    return kept.length >= 4 ? kept : detections;
  }

  /** 每次骨架推論後呼叫；kp 為 null 表示這幀看不到投手 */
  ingest(kp: PoseKeypoints | null, ts: number): PoseReleaseEvent | null {
    if (!kp || kp.length < 17) {
      this.samples.push({ ts, personSeen: false, rSpeed: 0, lSpeed: 0, rAbove: -1, lAbove: -1, kneeLift: 0 });
      this.lastKp = null;
      return this.trimAndDetect();
    }
    this.lastPersonSeenTs = ts;
    const ok = (i: number) => kp[i][2] > MIN_CONF;
    const midY = (a: number, b: number) => {
      const vals = [a, b].filter(ok).map((i) => kp[i][1]);
      return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
    };
    const shY = midY(KP.L_SHOULDER, KP.R_SHOULDER);
    const hpY = midY(KP.L_HIP, KP.R_HIP);
    const torso = shY !== null && hpY !== null ? Math.max(24, Math.abs(hpY - shY)) : 120;

    const rAbove = ok(KP.R_WRIST) && ok(KP.R_SHOULDER) ? (kp[KP.R_SHOULDER][1] - kp[KP.R_WRIST][1]) / torso : -1;
    const lAbove = ok(KP.L_WRIST) && ok(KP.L_SHOULDER) ? (kp[KP.L_SHOULDER][1] - kp[KP.L_WRIST][1]) / torso : -1;

    let kneeLift = 0;
    if (hpY !== null) {
      const knees = [KP.L_KNEE, KP.R_KNEE].filter(ok).map((i) => kp[i][1]);
      if (knees.length) {
        const highest = Math.min(...knees);
        kneeLift = Math.max(0, 1 - (highest - hpY) / Math.max(24, torso * 1.1));
      }
    }

    let rSpeed = 0;
    let lSpeed = 0;
    const prev = this.lastKp;
    if (prev && this.lastKpTs > 0) {
      const dt = Math.max(0.008, (ts - this.lastKpTs) / 1000);
      const sp = (w: number) =>
        kp[w][2] > MIN_CONF && prev[w][2] > MIN_CONF
          ? Math.min(18, Math.hypot(kp[w][0] - prev[w][0], kp[w][1] - prev[w][1]) / torso / dt)
          : 0;
      rSpeed = sp(KP.R_WRIST);
      lSpeed = sp(KP.L_WRIST);
    }
    this.lastKp = kp;
    this.lastKpTs = ts;
    this.samples.push({ ts, personSeen: true, rSpeed, lSpeed, rAbove, lAbove, kneeLift });
    return this.trimAndDetect();
  }

  private trimAndDetect(): PoseReleaseEvent | null {
    const last = this.samples[this.samples.length - 1];
    if (last) {
      this.samples = this.samples.filter((s) => last.ts - s.ts <= 4000);
      this.releases = this.releases.filter((r) => last.ts - r.releaseMs <= 8000);
    }
    const n = this.samples.length;
    if (n < 5) return null;

    let rE = 0;
    let lE = 0;
    for (const s of this.samples) {
      rE += s.rSpeed * (s.rAbove > 0.05 ? 1.2 : 0.3);
      lE += s.lSpeed * (s.lAbove > 0.05 ? 1.2 : 0.3);
    }
    const useR = rE >= lE;
    const speed = (i: number) => (useR ? this.samples[i].rSpeed : this.samples[i].lSpeed);
    const above = (i: number) => (useR ? this.samples[i].rAbove : this.samples[i].lAbove);
    const smooth = (i: number) => {
      const lo = Math.max(0, i - 1);
      const hi = Math.min(n - 1, i + 1);
      let sum = 0;
      for (let k = lo; k <= hi; k++) sum += speed(k);
      return sum / (hi - lo + 1);
    };
    const score = (i: number) => {
      const a = above(i);
      const s = smooth(i);
      if (a <= -0.25) return s * 0.2;
      return s * (1 + Math.max(-0.4, Math.min(1, a + 0.15)));
    };

    // 倒數第 2 個樣本為局部峰值（下一個樣本確認）
    const c = n - 2;
    const sc = score(c);
    if (sc < PitcherPoseGate.RELEASE_SCORE_THRESHOLD || sc < score(c - 1) || sc <= score(c + 1)) return null;
    const peakTs = this.samples[c].ts;
    const lastRel = this.releases[this.releases.length - 1];
    if (lastRel && peakTs - lastRel.releaseMs < PitcherPoseGate.MIN_RELEASE_GAP_MS) return null;

    // 準備動作守門
    let windup = 0;
    for (let i = 0; i < c - 1; i++) {
      const s = this.samples[i];
      if (!s.personSeen || peakTs - s.ts > PitcherPoseGate.WINDUP_LOOKBACK_MS) continue;
      if (s.kneeLift > 0.35 || smooth(i) > 0.8) windup++;
    }
    if (windup < PitcherPoseGate.MIN_WINDUP_SAMPLES) return null;

    const ev: PoseReleaseEvent = { releaseMs: peakTs, throwingHand: useR ? 'R' : 'L', score: sc };
    this.releases.push(ev);
    return ev;
  }
}

/**
 * 解析 YOLO11n-pose ONNX 輸出 [1, 56, N]（4 box + 1 conf + 17×3 kps），挑出投手（大、靠中央）。
 * 回傳像素座標（已從 letterbox 映射回原圖）。
 */
export function parsePitcherPose(
  data: Float32Array,
  anchors: number,
  map: { padX: number; padY: number; scale: number; origW: number; origH: number },
  minConf = 0.3,
): PoseKeypoints | null {
  const { padX, padY, scale, origW, origH } = map;
  let bestIdx = -1;
  let bestScore = -1;
  for (let i = 0; i < anchors; i++) {
    const conf = data[4 * anchors + i];
    if (conf < minConf) continue;
    const bw = data[2 * anchors + i] / scale;
    const bh = data[3 * anchors + i] / scale;
    const cx = (data[i] - padX) / scale;
    const cy = (data[anchors + i] - padY) / scale;
    if (bh / origH < 0.16 || cy / origH < 0.15 || cy / origH > 0.9) continue;
    const centerBias = Math.max(0.2, 1 - Math.abs(cx / origW - 0.48) * 1.4);
    const s = ((bw * bh) / (origW * origH)) * centerBias * conf;
    if (s > bestScore) {
      bestScore = s;
      bestIdx = i;
    }
  }
  if (bestIdx < 0) return null;
  const kps: [number, number, number][] = [];
  for (let k = 0; k < 17; k++) {
    const base = 5 + k * 3;
    kps.push([
      (data[base * anchors + bestIdx] - padX) / scale,
      (data[(base + 1) * anchors + bestIdx] - padY) / scale,
      data[(base + 2) * anchors + bestIdx],
    ]);
  }
  return kps;
}
