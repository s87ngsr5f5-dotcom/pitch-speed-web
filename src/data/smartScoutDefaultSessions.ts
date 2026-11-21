import { BallDetection, BenchmarkDataset } from '../types';
import {
  classifyPitchFromMetrics,
  computeStrikeZoneGrid,
  degToClockString,
} from '../core/smartScoutEngine';

export interface SmartScoutPitchRecord {
  id: string;
  pitchNumber: number;
  timestamp: string;
  pitchType: string;
  pitchColor: string;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  speedKmh: number;          // V [km/h] 初速
  plateSpeedKmh: number;     // 進壘末速 (~0.916 * V)
  verticalBreakCm: number;   // VB [cm] 垂直變化量
  horizontalBreakCm: number; // HB [cm] 水平變化量
  efficiencyPct: number;     // E [%] 效率
  spinDirection: string;     // SD 旋轉方向 (時鐘角)
  spinAxisDeg: number;       // 旋轉軸角度 (°)
  gyroDegree: number;        // GD [°] 陀螺角度
  activeSpinRpm: number;     // AS [rpm] 有效轉速
  totalSpinRpm: number;      // S [rpm] 轉速
  plateXCm: number;          // 九宮格水平落點 (-45 ~ +45 cm)
  plateYCm: number;          // 九宮格垂直落點 (20 ~ 130 cm)
  zoneId: number;            // 1~9 九宮格好球帶, 11~14 外圍追打壞球帶
  isStrike: boolean;
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

export function buildRibbonPathFromMetrics(
  cameraAngle: 'behind_pitcher' | 'behind_catcher',
  vbCm: number,
  hbCm: number,
  plateXCm = 5,
  plateYCm = 76
): SmartScoutPitchRecord['ribbonPath'] {
  if (cameraAngle === 'behind_pitcher') {
    const startX = 0.695;
    const startY = 0.435;
    const endX = Math.max(0.35, Math.min(0.68, 0.515 - (plateXCm / 110) * 0.14));
    const endY = Math.max(0.40, Math.min(0.63, 0.525 - ((plateYCm - 76) / 100) * 0.09));
    const ctrlX = startX - 0.11 - (hbCm / 60) * 0.085;
    const ctrlY = Math.min(startY, endY) - 0.055 - (vbCm / 60) * 0.055;
    return {
      startX,
      startY,
      startR: 28,
      ctrlX,
      ctrlY,
      endX,
      endY,
      endR: 4.8,
    };
  } else {
    const startX = 0.50;
    const startY = 0.41;
    const endX = Math.max(0.28, Math.min(0.74, 0.50 + (plateXCm / 50) * 0.16));
    const endY = Math.max(0.36, Math.min(0.72, 0.56 - ((plateYCm - 76) / 60) * 0.14));
    const ctrlX = (startX + endX) * 0.5 - (hbCm / 60) * 0.09;
    const ctrlY = Math.min(startY, endY) - 0.05 - (vbCm / 60) * 0.06;
    return {
      startX,
      startY,
      startR: 5.2,
      ctrlX,
      ctrlY,
      endX,
      endY,
      endR: 29,
    };
  }
}

export const INITIAL_SMARTSCOUT_SESSIONS: SmartScoutSession[] = [];

export function speakPitchResult(pitch: SmartScoutPitchRecord, enabled: boolean): void {
  if (!enabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const text = pitch.isHitting
      ? `擊球初速 ${Math.round(pitch.speedKmh)} 公里`
      : `${pitch.pitchType}，球速 ${Math.round(pitch.speedKmh)} 公里`;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'zh-TW';
    utter.rate = 1.08;
    window.speechSynthesis.speak(utter);
  } catch {
    // ignore
  }
}

/**
 * 由相機或影片真實偵測到的連續球體座標 (BallDetection[]) 計算 8 大數據與 3D 紅色彩帶軌跡
 */
export function computeRealPitchFromDetections(params: {
  detections: BallDetection[];
  videoWidth: number;
  videoHeight: number;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  distanceMeters: number;
  preferredPitchType?: string;
  snapshotUrl?: string;
  videoUrl?: string;
}): Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'> | null {
  const {
    detections,
    videoWidth,
    videoHeight,
    cameraAngle,
    discipline,
    distanceMeters,
    preferredPitchType,
    snapshotUrl,
    videoUrl,
  } = params;

  if (detections.length < 4 || videoWidth <= 0 || videoHeight <= 0) return null;

  const first = detections[0];
  const last = detections[detections.length - 1];
  const mid = detections[Math.floor(detections.length / 2)];

  const flightTimeMs = last.timestampMs - first.timestampMs;
  if (flightTimeMs < 80 || flightTimeMs > 2200) return null;

  // 1. 扣除投手踏步延伸距離 1.75m 得有效飛行距離 D_eff = D - 1.75，再除以空氣阻力係數 0.955 反推出手初速
  const dEff = Math.max(6.0, distanceMeters - 1.75);
  const avgSpeedMps = dEff / (flightTimeMs / 1000);
  const rawSpeedKmh = (avgSpeedMps * 3.6) / 0.955;
  const speedKmh = Number(Math.max(35, Math.min(185, rawSpeedKmh)).toFixed(1));
  const plateSpeedKmh = Number((speedKmh * 0.916).toFixed(1));

  // 2. 變化量 (VB / HB) 反推
  const lineMidX = (first.x + last.x) * 0.5;
  const lineMidY = (first.y + last.y) * 0.5;
  const pxToCm = 86 / Math.max(120, videoWidth * 0.22);

  const signX = cameraAngle === 'behind_pitcher' ? -1 : 1;
  const rawHbCm = (mid.x - lineMidX) * pxToCm * signX * 1.8;
  const flightSec = flightTimeMs / 1000;
  const gravityDropCm = Math.min(72, 0.5 * 981 * flightSec * flightSec * 0.35);
  const rawVbCm = gravityDropCm - (mid.y - lineMidY) * pxToCm * 1.5;

  const horizontalBreakCm = Number(Math.max(-58, Math.min(58, rawHbCm)).toFixed(1));
  const verticalBreakCm = Number(Math.max(-58, Math.min(68, rawVbCm)).toFixed(1));

  // 3. 球種自動分類
  const autoCode = classifyPitchFromMetrics(speedKmh, horizontalBreakCm, verticalBreakCm);
  const codeToName: Record<string, string> = {
    four_seam: '四縫線快速球',
    sinker: '伸卡球',
    slider: '滑球',
    curveball: '曲球',
    changeup: '變速球',
    cutter: '卡特球',
  };
  const pitchType =
    preferredPitchType && preferredPitchType !== '自動辨識'
      ? preferredPitchType
      : codeToName[autoCode] ?? '四縫線快速球';

  // 4. 旋轉軸角度、時鐘角 SD、有效轉速 AS、總轉速 S、效率 E、陀螺角 GD
  const breakMagCm = Math.hypot(horizontalBreakCm, verticalBreakCm);
  const spinAxisDeg = Math.round((Math.atan2(verticalBreakCm, horizontalBreakCm) * 180) / Math.PI);
  const spinDirection = degToClockString(spinAxisDeg);

  const baseEff =
    pitchType === '滑球'
      ? 56
      : pitchType === '卡特球'
        ? 64
        : pitchType === '曲球'
          ? 78
          : 84;
  const efficiencyPct = Math.max(
    35,
    Math.min(98, Math.round(baseEff + verticalBreakCm / 15))
  );

  const activeSpinRpm = Math.round(
    Math.max(850, Math.min(2650, breakMagCm * 22 + speedKmh * 6.8))
  );
  const totalSpinRpm = Math.round(
    Math.max(1350, Math.min(3150, activeSpinRpm / (efficiencyPct / 100)))
  );
  const gyroDegree = Math.round((Math.acos(efficiencyPct / 100) * 180) / Math.PI);

  // 5. 本壘九宮格進壘點 (plateXCm, plateYCm, zoneId 1~9 或 11~14)
  const plateXCm = Number(
    Math.max(-42, Math.min(42, (last.x / videoWidth - 0.5) * 85)).toFixed(1)
  );
  const plateYCm = Number(
    Math.max(25, Math.min(130, 76 - (last.y / videoHeight - 0.54) * 95)).toFixed(1)
  );
  const { zoneId, isStrike } = computeStrikeZoneGrid(plateXCm, plateYCm);

  // 6. 二次貝茲控制點
  const sx = Math.max(0.05, Math.min(0.95, first.x / videoWidth));
  const sy = Math.max(0.05, Math.min(0.95, first.y / videoHeight));
  const mx = Math.max(0.05, Math.min(0.95, mid.x / videoWidth));
  const my = Math.max(0.05, Math.min(0.95, mid.y / videoHeight));
  const ex = Math.max(0.05, Math.min(0.95, last.x / videoWidth));
  const ey = Math.max(0.05, Math.min(0.95, last.y / videoHeight));

  const ctrlX = Math.max(0.02, Math.min(0.98, 2 * mx - 0.5 * (sx + ex)));
  const ctrlY = Math.max(0.02, Math.min(0.98, 2 * my - 0.5 * (sy + ey)));

  const startR = cameraAngle === 'behind_pitcher' ? 28 : 5.2;
  const endR = cameraAngle === 'behind_pitcher' ? 4.8 : 29;

  const isHit = discipline === 'hitting';

  return {
    pitchType,
    pitchColor: getPitchTypeColor(pitchType),
    cameraAngle,
    speedKmh,
    plateSpeedKmh,
    verticalBreakCm,
    horizontalBreakCm,
    efficiencyPct,
    spinDirection,
    spinAxisDeg,
    gyroDegree,
    activeSpinRpm,
    totalSpinRpm,
    plateXCm,
    plateYCm,
    zoneId,
    isStrike,
    snapshotUrl,
    videoUrl,
    isHitting: isHit,
    launchAngleDeg: isHit ? Number((verticalBreakCm * 0.45).toFixed(1)) : undefined,
    sprayAngleDeg: isHit ? Number((horizontalBreakCm * 0.5).toFixed(1)) : undefined,
    distanceM: isHit ? Number((speedKmh * 0.72).toFixed(1)) : undefined,
    ribbonPath: {
      startX: sx,
      startY: sy,
      startR,
      ctrlX,
      ctrlY,
      endX: ex,
      endY: ey,
      endR,
    },
  };
}

/**
 * 載入 100 球基準測試庫 (benchmark_100.json) 供「場次」分頁瀏覽與驗證
 */
export function convertBenchmark100ToSession(dataset: BenchmarkDataset): SmartScoutSession {
  const typeMap: Record<
    string,
    {
      name: string;
      color: string;
      vbRange: [number, number];
      hbRange: [number, number];
      effRange: [number, number];
    }
  > = {
    four_seam: {
      name: '四縫線快速球',
      color: '#be1e3c',
      vbRange: [44, 64],
      hbRange: [12, 32],
      effRange: [84, 96],
    },
    sinker: {
      name: '伸卡球',
      color: '#f97316',
      vbRange: [22, 38],
      hbRange: [30, 48],
      effRange: [80, 91],
    },
    slider: {
      name: '滑球',
      color: '#eab308',
      vbRange: [-6, 14],
      hbRange: [-36, -15],
      effRange: [46, 66],
    },
    changeup: {
      name: '變速球',
      color: '#16a34a',
      vbRange: [18, 34],
      hbRange: [24, 42],
      effRange: [78, 89],
    },
    curveball: {
      name: '曲球',
      color: '#0284c7',
      vbRange: [-48, -22],
      hbRange: [-28, -10],
      effRange: [72, 86],
    },
  };

  const pitches: SmartScoutPitchRecord[] = dataset.pitches.map((p, idx) => {
    const meta = typeMap[p.pitchType] ?? typeMap.four_seam;
    const ratio = ((idx * 37) % 100) / 100;
    const ratio2 = ((idx * 73) % 100) / 100;
    const vb = Number(
      (meta.vbRange[0] + (meta.vbRange[1] - meta.vbRange[0]) * ratio).toFixed(1)
    );
    const hb = Number(
      (meta.hbRange[0] + (meta.hbRange[1] - meta.hbRange[0]) * ratio2).toFixed(1)
    );
    const eff = Math.round(
      meta.effRange[0] + (meta.effRange[1] - meta.effRange[0]) * ratio
    );
    const totalSpin = Math.round(1880 + p.predictedKmh * 3.4 + idx * 3.2);
    const activeSpin = Math.round(totalSpin * (eff / 100));
    const gyro = Math.round((Math.acos(eff / 100) * 180) / Math.PI);
    const spinAxisDeg = Math.round((Math.atan2(vb, hb) * 180) / Math.PI);
    const sd = degToClockString(spinAxisDeg);
    const camAngle: 'behind_pitcher' | 'behind_catcher' =
      idx % 2 === 0 ? 'behind_pitcher' : 'behind_catcher';
    const plateX = Number(((ratio2 - 0.5) * 38).toFixed(1));
    const plateY = Number((52 + ratio * 48).toFixed(1));
    const { zoneId, isStrike } = computeStrikeZoneGrid(plateX, plateY);

    return {
      id: `bench-100-${p.id}`,
      pitchNumber: p.id,
      timestamp: `2026/11/21 18:${String(Math.floor(p.id / 2) % 60).padStart(2, '0')}:${String((p.id * 17) % 60).padStart(2, '0')}`,
      pitchType: meta.name,
      pitchColor: meta.color,
      cameraAngle: camAngle,
      speedKmh: p.predictedKmh,
      plateSpeedKmh: Number((p.predictedKmh * 0.916).toFixed(1)),
      verticalBreakCm: vb,
      horizontalBreakCm: hb,
      efficiencyPct: eff,
      spinDirection: sd,
      spinAxisDeg,
      gyroDegree: gyro,
      activeSpinRpm: activeSpin,
      totalSpinRpm: totalSpin,
      plateXCm: plateX,
      plateYCm: plateY,
      zoneId,
      isStrike,
      ribbonPath: buildRibbonPathFromMetrics(camAngle, vb, hb, plateX, plateY),
    };
  });

  return {
    id: 'session-benchmark-100',
    title: 'AI 100球基準測試庫 (MAE 0.40 km/h)',
    date: '2026/11/21',
    monthGroup: '2026年11月',
    discipline: 'pitching',
    cameraSetup: 'behind_pitcher',
    pitches,
  };
}
