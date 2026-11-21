import { BallDetection, TunnelBoxConfig } from '../types';
import { estimateAutomaticPitchFlight, estimatePitchFlight } from '../core/smartScoutEngine';

export interface SmartScoutPitchRecord {
  id: string;
  pitchNumber: number;
  timestamp: string;
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
  const pitchType = preferredPitchType || '未標記';
  const behindPitcher = cameraAngle === 'behind_pitcher';

  return {
    pitchType,
    pitchColor: getPitchTypeColor(pitchType),
    cameraAngle,
    speedKmh,
    measurement: { flightTimeMs, detectionCount, automaticEndpoints },
    snapshotUrl,
    videoUrl,
    ribbonPath: {
      startX,
      startY,
      startR: behindPitcher ? 28 : 5.2,
      ctrlX: Math.max(0, Math.min(1, 2 * mid.x / videoWidth - (startX + endX) / 2)),
      ctrlY: Math.max(0, Math.min(1, 2 * mid.y / videoHeight - (startY + endY) / 2)),
      endX,
      endY,
      endR: behindPitcher ? 4.8 : 29,
    },
  };
}
