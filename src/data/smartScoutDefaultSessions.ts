import { BallDetection, TunnelBoxConfig } from '../types';
import {
  buildSmartScoutMetrics,
  estimateAutomaticPitchFlight,
  estimatePitchFlight,
} from '../core/smartScoutEngine';

export interface SmartScoutPitchRecord {
  id: string;
  pitchNumber: number;
  timestamp: string;
  recordedDate?: string;     // YYYY-MM-DD 實測日期
  videoClipId?: string;      // IndexedDB 內儲存的單球影片 ID
  pitchType: string;
  pitchColor: string;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  speedKmh: number;          // V [km/h] 初速
  plateSpeedKmh?: number;
  verticalBreakCm?: number;
  horizontalBreakCm?: number;
  efficiencyPct?: number;
  spinDirection?: string;
  spinAxisDeg?: number;
  gyroDegree?: number;
  activeSpinRpm?: number;
  totalSpinRpm?: number;
  plateXCm?: number;
  plateYCm?: number;
  zoneId?: number;
  isStrike?: boolean;
  measurement?: { flightTimeMs: number; detectionCount: number; automaticEndpoints?: boolean };
  snapshotUrl?: string;      // 實測當下擷取之相機真實畫面
  videoUrl?: string;         // 使用者上傳之真實影片
  ribbonPath: {
    startX: number;
    startY: number;
    startR: number;
    ctrlX: number;
    ctrlY: number;
    endX: number;
    endY: number;
    endR: number;
  };
  isHitting?: boolean;
  launchAngleDeg?: number;
  sprayAngleDeg?: number;
  distanceM?: number;
}

export interface SmartScoutSession {
  id: string;
  title: string;
  date: string;
  monthGroup: string;
  discipline: 'pitching' | 'hitting';
  cameraSetup: 'behind_pitcher' | 'behind_catcher';
  pitches: SmartScoutPitchRecord[];
}

export const SMARTSCOUT_PITCH_TYPES: Array<{
  name: string;
  color: string;
}> = [
  { name: '未標記', color: '#475569' },
  { name: '四縫線快速球', color: '#be1e3c' },
  { name: '伸卡球', color: '#f97316' },
  { name: '滑球', color: '#eab308' },
  { name: '曲球', color: '#0284c7' },
  { name: '變速球', color: '#16a34a' },
  { name: '卡特球', color: '#9333ea' },
  { name: '指叉球', color: '#0d9488' },
];

export function getPitchTypeColor(pitchType: string): string {
  const found = SMARTSCOUT_PITCH_TYPES.find((p) => p.name === pitchType);
  return found ? found.color : '#be1e3c';
}

export interface RibbonSample {
  x: number;
  y: number;
  r: number;
}

/**
 * 二次貝茲 3D 漸變紅色彗星彩帶取樣演算法
 * r(t) = r_start + (r_end - r_start) * t^0.85
 */
export function sampleQuadraticRibbon(
  path: SmartScoutPitchRecord['ribbonPath'],
  width: number,
  height: number,
  maxT: number,
  steps = 46
): RibbonSample[] {
  const sx = path.startX * width;
  const sy = path.startY * height;
  const cx = path.ctrlX * width;
  const cy = path.ctrlY * height;
  const ex = path.endX * width;
  const ey = path.endY * height;

  const scale = Math.min(width, height) / 420;
  const sr = path.startR * scale;
  const er = path.endR * scale;

  const out: RibbonSample[] = [];
  const count = Math.max(2, Math.round(steps * Math.max(0.05, maxT)));
  for (let i = 0; i <= count; i++) {
    const t = (i / count) * maxT;
    const inv = 1 - t;
    const x = inv * inv * sx + 2 * inv * t * cx + t * t * ex;
    const y = inv * inv * sy + 2 * inv * t * cy + t * t * ey;
    const r = sr + (er - sr) * Math.pow(t, 0.85);
    out.push({ x, y, r });
  }
  return out;
}

/**
 * 沿曲線切線計算左右法向量 (nx, ny) 圍出平滑錐形多邊形 (Tapered Polygon)
 */
export function buildTaperedRibbonPolygon(samples: RibbonSample[]): string {
  if (samples.length < 2) return '';
  const leftPts: Array<{ x: number; y: number }> = [];
  const rightPts: Array<{ x: number; y: number }> = [];

  for (let i = 0; i < samples.length; i++) {
    const curr = samples[i];
    const prev = samples[Math.max(0, i - 1)];
    const next = samples[Math.min(samples.length - 1, i + 1)];
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;

    leftPts.push({ x: curr.x + nx * curr.r, y: curr.y + ny * curr.r });
    rightPts.push({ x: curr.x - nx * curr.r, y: curr.y - ny * curr.r });
  }

  const all = [...leftPts, ...rightPts.reverse()];
  return all.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

export const INITIAL_SMARTSCOUT_SESSIONS: SmartScoutSession[] = [];

export function speakPitchResult(pitch: SmartScoutPitchRecord, enabled: boolean): void {
  if (!enabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const text = pitch.isHitting
      ? `擊球初速 ${Math.round(pitch.speedKmh)} 公里`
      : `${pitch.pitchType}，預估球速 ${Math.round(pitch.speedKmh)} 公里每小時`;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'zh-TW';
    utter.rate = 1.08;
    window.speechSynthesis.speak(utter);
  } catch {
    // ignore
  }
}

export function classifyPitchTypeFromTrajectory(params: {
  speedKmh: number;
  horizontalBreakCm: number;
  verticalBreakCm: number;
}): string {
  const { speedKmh, horizontalBreakCm, verticalBreakCm } = params;
  const absHb = Math.abs(horizontalBreakCm);

  // 曲球：球速較慢且明顯下墜
  if (speedKmh < 112 && verticalBreakCm <= -12) return '曲球';
  // 滑球：橫向位移大且速度中等
  if (absHb >= 16 && verticalBreakCm <= 14 && speedKmh < 132) return '滑球';
  // 指叉球：中高球速但急劇下墜、橫向位移小
  if (verticalBreakCm <= -10 && absHb < 12 && speedKmh >= 112) return '指叉球';
  // 變速球：球速較慢、手臂側微幅偏移並伴隨下墜
  if (speedKmh < 118 && horizontalBreakCm >= 8 && verticalBreakCm < 16) return '變速球';
  // 伸卡球：高球速、手臂側位移明顯且縱向較低
  if (speedKmh >= 118 && horizontalBreakCm >= 14 && verticalBreakCm < 24) return '伸卡球';
  // 卡特球：高球速、手套側小幅切削
  if (speedKmh >= 118 && horizontalBreakCm <= -10 && verticalBreakCm >= 8) return '卡特球';
  // 預設四縫線快速球
  return '四縫線快速球';
}

/** Create a record only when the track reaches both checkpoints. */
export function computeRealPitchFromDetections(params: {
  detections: BallDetection[];
  videoWidth: number;
  videoHeight: number;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  distanceMeters: number;
  checkpoints?: TunnelBoxConfig;
  preferredPitchType?: string;
  snapshotUrl?: string;
  videoUrl?: string;
  videoClipId?: string;
}): Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'> | null {
  if (params.discipline !== 'pitching') return null;
  const automaticEndpoints = !params.checkpoints;
  const result = automaticEndpoints ? estimateAutomaticPitchFlight(params) : estimatePitchFlight(params);
  if (!result) return null;

  const { videoWidth, videoHeight, cameraAngle, preferredPitchType, snapshotUrl, videoUrl } = params;
  const { track, speedKmh, flightTimeMs, detectionCount } = result;
  const first = track[0];
  const last = track[track.length - 1];
  const mid = track[Math.floor(track.length / 2)];
  const startX = first.x / videoWidth;
  const startY = first.y / videoHeight;
  const endX = last.x / videoWidth;
  const endY = last.y / videoHeight;
  const behindPitcher = cameraAngle === 'behind_pitcher';

  const lineMidX = (first.x + last.x) * 0.5;
  const lineMidY = (first.y + last.y) * 0.5;
  const pxToCm = 86 / Math.max(110, videoWidth * 0.2);
  const signX = behindPitcher ? -1 : 1;
  const rawHbCm = (mid.x - lineMidX) * pxToCm * signX * 1.8;
  const flightSec = flightTimeMs / 1000;
  const gravityDropCm = Math.min(72, 0.5 * 981 * flightSec * flightSec * 0.35);
  const rawVbCm = gravityDropCm - (mid.y - lineMidY) * pxToCm * 1.5;

  const horizontalBreakCm = Number(Math.max(-58, Math.min(58, rawHbCm)).toFixed(1));
  const verticalBreakCm = Number(Math.max(-58, Math.min(68, rawVbCm)).toFixed(1));
  const plateXCm = Number(Math.max(-42, Math.min(42, (endX - 0.48) * 140)).toFixed(1));
  const plateYCm = Number(Math.max(25, Math.min(130, 76 - (endY - 0.50) * 120)).toFixed(1));

  const metrics = buildSmartScoutMetrics({
    speedKmh,
    horizontalBreakCm,
    verticalBreakCm,
    plateXCm,
    plateYCm,
  });

  const pitchType =
    preferredPitchType && preferredPitchType !== '未標記'
      ? preferredPitchType
      : classifyPitchTypeFromTrajectory({
          speedKmh,
          horizontalBreakCm,
          verticalBreakCm,
        });

  const now = new Date();
  const recordedDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;

  const ctrlX = Math.max(0, Math.min(1, 2 * mid.x / videoWidth - (startX + endX) / 2));
  const ctrlY = Math.max(0, Math.min(1, 2 * mid.y / videoHeight - (startY + endY) / 2));
  const normSpan = Math.max(
    Math.hypot(endX - startX, endY - startY),
    Math.hypot(mid.x / videoWidth - startX, mid.y / videoHeight - startY) * 1.6
  );
  const radiusScale = Math.max(0.45, Math.min(1.0, normSpan / 0.12));

  return {
    recordedDate,
    videoClipId: params.videoClipId,
    pitchType,
    pitchColor: getPitchTypeColor(pitchType),
    cameraAngle,
    speedKmh,
    plateSpeedKmh: metrics.plateSpeedKmh,
    verticalBreakCm: metrics.verticalBreakCm,
    horizontalBreakCm: metrics.horizontalBreakCm,
    efficiencyPct: metrics.spinEfficiencyPct,
    spinDirection: metrics.spinAxisClock,
    spinAxisDeg: metrics.spinAxisDeg,
    gyroDegree: Math.round(metrics.gyroAngleDeg),
    activeSpinRpm: metrics.activeSpinRpm,
    totalSpinRpm: metrics.spinRateRpm,
    plateXCm: metrics.plateXCm,
    plateYCm: metrics.plateYCm,
    zoneId: metrics.zoneId,
    isStrike: metrics.isStrike,
    measurement: { flightTimeMs, detectionCount, automaticEndpoints },
    snapshotUrl,
    videoUrl,
    ribbonPath: {
      startX,
      startY,
      startR: Number(((behindPitcher ? 28 : 5.2) * radiusScale).toFixed(1)),
      ctrlX,
      ctrlY,
      endX,
      endY,
      endR: Number(((behindPitcher ? 4.8 : 29) * radiusScale).toFixed(1)),
    },
  };
}
