import { BallDetection } from '../types';
import { classifyPitchFromMetrics, degToClockString } from '../core/smartScoutEngine';

export interface SmartScoutPitchRecord {
  id: string;
  pitchNumber: number;
  timestamp: string;
  pitchType: string;
  pitchColor: string;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  speedKmh: number;          // V [km/h] 初速
  verticalBreakCm: number;   // VB [cm] 垂直變化量
  horizontalBreakCm: number; // HB [cm] 水平變化量
  efficiencyPct: number;     // E [%] 效率
  spinDirection: string;     // SD 旋轉方向
  gyroDegree: number;        // GD [°] 陀螺角度
  activeSpinRpm: number;     // AS [rpm] 有效轉速
  totalSpinRpm: number;      // S [rpm] 轉速
  plateXCm: number;          // 九宮格水平落點 (-45 ~ +45 cm)
  plateYCm: number;          // 九宮格垂直落點 (20 ~ 130 cm)
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
  { name: '二縫線伸卡球', color: '#ea580c' },
  { name: '滑球', color: '#ca8a04' },
  { name: '曲球', color: '#0284c7' },
  { name: '變速球', color: '#059669' },
  { name: '卡特球', color: '#7c3aed' },
  { name: '指叉球', color: '#0d9488' },
];

export function getPitchTypeColor(pitchType: string): string {
  const found = SMARTSCOUT_PITCH_TYPES.find((p) => p.name === pitchType);
  return found ? found.color : '#be1e3c';
}

// 預設為空陣列：不預載任何假投球數據或範例影片，所有數據皆由相機/影片實測產生
export const INITIAL_SMARTSCOUT_SESSIONS: SmartScoutSession[] = [];

export function speakPitchResult(pitch: SmartScoutPitchRecord, enabled: boolean): void {
  if (!enabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const text = pitch.isHitting
      ? `擊球初速 ${pitch.speedKmh.toFixed(1)} 公里`
      : `初速 ${pitch.speedKmh.toFixed(1)} 公里，${pitch.pitchType}，轉速 ${pitch.totalSpinRpm} 轉`;
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

  // 1. 根據飛行時間、畫面位移比例與球體視覺直徑縮放比推算實際飛行距離與初速
  const pixelSpan = Math.hypot(last.x - first.x, last.y - first.y);
  const diag = Math.hypot(videoWidth, videoHeight);
  const spanRatio = Math.min(1.0, Math.max(0.35, (pixelSpan / diag) * 2.4));
  const effectiveDistM = Math.max(5.0, (distanceMeters - 1.7) * spanRatio);

  const avgSpeedMps = effectiveDistM / (flightTimeMs / 1000);
  const rawSpeedKmh = (avgSpeedMps * 3.6) / 0.955;
  const speedKmh = Number(Math.max(35, Math.min(185, rawSpeedKmh)).toFixed(1));

  // 2. 由中點偏離首尾連線之像素曲率與重力拋物線推算水平變化量 (HB) 與垂直變化量 (VB)
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

  // 3. 自動分類或套用球種
  const autoCode = classifyPitchFromMetrics(speedKmh, horizontalBreakCm, verticalBreakCm);
  const codeToName: Record<string, string> = {
    four_seam: '四縫線快速球',
    sinker: '二縫線伸卡球',
    slider: '滑球',
    curveball: '曲球',
    changeup: '變速球',
    cutter: '卡特球',
  };
  const pitchType =
    preferredPitchType && preferredPitchType !== '自動辨識'
      ? preferredPitchType
      : codeToName[autoCode] ?? '四縫線快速球';

  // 4. 由位移向量與球速反推旋轉軸 (SD)、有效轉速 (AS)、總轉速 (S)、旋轉效率 (E) 與陀螺角 (GD)
  const breakMagCm = Math.hypot(horizontalBreakCm, verticalBreakCm);
  const axisDeg = Math.round((Math.atan2(verticalBreakCm, horizontalBreakCm) * 180) / Math.PI);
  const spinDirection = degToClockString(axisDeg);

  const baseEff =
    pitchType === '滑球'
      ? 56
      : pitchType === '卡特球'
        ? 64
        : pitchType === '曲球'
          ? 78
          : 83;
  const efficiencyPct = Math.max(
    35,
    Math.min(98, Math.round(baseEff + (verticalBreakCm / 15)))
  );

  const activeSpinRpm = Math.round(
    Math.max(850, Math.min(2650, breakMagCm * 22 + speedKmh * 6.8))
  );
  const totalSpinRpm = Math.round(
    Math.max(1350, Math.min(3150, activeSpinRpm / (efficiencyPct / 100)))
  );
  const gyroDegree = Math.round((Math.acos(efficiencyPct / 100) * 180) / Math.PI);

  // 5. 進壘點相對於畫面好球帶中心之本壘座標
  const plateXCm = Number(
    Math.max(-42, Math.min(42, ((last.x / videoWidth - 0.5) * 85))).toFixed(1)
  );
  const plateYCm = Number(
    Math.max(25, Math.min(130, 76 - (last.y / videoHeight - 0.54) * 95)).toFixed(1)
  );

  // 6. 將真實偵測到的首、中、尾像素點反算為二次貝茲曲線控制點，使紅色彩帶完全貼合真實球路
  const sx = Math.max(0.05, Math.min(0.95, first.x / videoWidth));
  const sy = Math.max(0.05, Math.min(0.95, first.y / videoHeight));
  const mx = Math.max(0.05, Math.min(0.95, mid.x / videoWidth));
  const my = Math.max(0.05, Math.min(0.95, mid.y / videoHeight));
  const ex = Math.max(0.05, Math.min(0.95, last.x / videoWidth));
  const ey = Math.max(0.05, Math.min(0.95, last.y / videoHeight));

  // 二次貝茲在 t=0.5 經過 (mx, my) 的控制點公式：C = 2*M - 0.5*(S + E)
  const ctrlX = Math.max(0.02, Math.min(0.98, 2 * mx - 0.5 * (sx + ex)));
  const ctrlY = Math.max(0.02, Math.min(0.98, 2 * my - 0.5 * (sy + ey)));

  const startR = cameraAngle === 'behind_pitcher' ? 28 : 5.5;
  const endR = cameraAngle === 'behind_pitcher' ? 5.0 : 28;

  const isHit = discipline === 'hitting';

  return {
    pitchType,
    pitchColor: getPitchTypeColor(pitchType),
    cameraAngle,
    speedKmh,
    verticalBreakCm,
    horizontalBreakCm,
    efficiencyPct,
    spinDirection,
    gyroDegree,
    activeSpinRpm,
    totalSpinRpm,
    plateXCm,
    plateYCm,
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
