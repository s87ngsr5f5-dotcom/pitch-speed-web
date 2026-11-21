import { BenchmarkDataset } from '../types';

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
  spinDirection: string;     // SD 旋轉方向 (時鐘 01:11)
  gyroDegree: number;        // GD [°] 陀螺角度
  activeSpinRpm: number;     // AS [rpm] 有效轉速
  totalSpinRpm: number;      // S [rpm] 轉速
  plateXCm: number;          // 九宮格水平落點 (-45 ~ +45 cm)
  plateYCm: number;          // 九宮格垂直落點 (20 ~ 130 cm)
  videoUrl?: string;
  // 3D 紅色彩帶軌跡控制點 (正規化 0~1: release 出手點 -> mid 弧頂 -> target 進壘點)
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
  // 若為打擊紀錄則附加打擊指標
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

/**
 * 根據投手正後方或正前方(捕手視角)與 VB/HB 位移量，產生 SmartScout 經典 3D 紅色漸變彗星彩帶控制點
 */
export function buildRibbonPathFromMetrics(
  cameraAngle: 'behind_pitcher' | 'behind_catcher',
  vbCm: number,
  hbCm: number,
  plateXCm = 5,
  plateYCm = 76
): SmartScoutPitchRecord['ribbonPath'] {
  if (cameraAngle === 'behind_pitcher') {
    // 鏡頭在投手正後方（如範例影片）：近端右上方投手出手時彩帶較寬(startR=29)，往左上方彎曲後向下收斂至遠端捕手手套(endR=4.5)
    const startX = 0.705;
    const startY = 0.445;
    const endX = Math.max(0.36, Math.min(0.68, 0.535 - (plateXCm / 120) * 0.14));
    const endY = Math.max(0.42, Math.min(0.62, 0.522 - ((plateYCm - 76) / 100) * 0.08));
    const ctrlX = startX - 0.12 - (hbCm / 60) * 0.08;
    const ctrlY = Math.min(startY, endY) - 0.055 - (vbCm / 60) * 0.055;
    return {
      startX,
      startY,
      startR: 29,
      ctrlX,
      ctrlY,
      endX,
      endY,
      endR: 4.5,
    };
  } else {
    // 鏡頭在投手正前方（捕手後方視角）：遠端投手出手時較細(startR=5.5)，飛向近端好球帶時放大(endR=31)
    const startX = 0.51;
    const startY = 0.43;
    const endX = Math.max(0.28, Math.min(0.74, 0.50 + (plateXCm / 50) * 0.16));
    const endY = Math.max(0.36, Math.min(0.72, 0.55 - ((plateYCm - 76) / 60) * 0.14));
    const ctrlX = (startX + endX) * 0.5 - (hbCm / 60) * 0.09;
    const ctrlY = Math.min(startY, endY) - 0.05 - (vbCm / 60) * 0.06;
    return {
      startX,
      startY,
      startR: 5.5,
      ctrlX,
      ctrlY,
      endX,
      endY,
      endR: 30,
    };
  }
}

export function speakPitchResult(pitch: SmartScoutPitchRecord, enabled: boolean): void {
  if (!enabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const text = pitch.isHitting
      ? `擊球初速 ${pitch.speedKmh.toFixed(1)} 公里，仰角 ${pitch.launchAngleDeg ?? 18} 度，飛行距離 ${pitch.distanceM ?? 85} 公尺`
      : `初速 ${pitch.speedKmh.toFixed(1)} 公里，${pitch.pitchType}，垂直變化量 ${pitch.verticalBreakCm.toFixed(1)} 公分，轉速 ${pitch.totalSpinRpm} 轉`;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'zh-TW';
    utter.rate = 1.08;
    window.speechSynthesis.speak(utter);
  } catch {
    // ignore speech error on unsupported browsers
  }
}

export const INITIAL_SMARTSCOUT_SESSIONS: SmartScoutSession[] = [
  {
    id: 'session-2026-08-26-pitch',
    title: '2026/08/26 投球',
    date: '2026/08/26',
    monthGroup: '2026年8月',
    discipline: 'pitching',
    cameraSetup: 'behind_pitcher',
    pitches: [
      {
        id: 'p-0826-1',
        pitchNumber: 1,
        timestamp: '2026/08/26 17:44:38',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 107.0,
        verticalBreakCm: 52.7,
        horizontalBreakCm: 37.7,
        efficiencyPct: 77,
        spinDirection: '01:11',
        gyroDegree: 39,
        activeSpinRpm: 1685,
        totalSpinRpm: 2178,
        plateXCm: 8.5,
        plateYCm: 82.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 52.7, 37.7, 8.5, 82.0),
      },
      {
        id: 'p-0826-2',
        pitchNumber: 2,
        timestamp: '2026/08/26 17:46:16',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 103.2,
        verticalBreakCm: 64.0,
        horizontalBreakCm: 22.7,
        efficiencyPct: 86,
        spinDirection: '00:39',
        gyroDegree: 31,
        activeSpinRpm: 1754,
        totalSpinRpm: 2044,
        plateXCm: -4.2,
        plateYCm: 88.5,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 64.0, 22.7, -4.2, 88.5),
      },
      {
        id: 'p-0826-3',
        pitchNumber: 3,
        timestamp: '2026/08/26 17:47:10',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 104.8,
        verticalBreakCm: 66.2,
        horizontalBreakCm: 25.1,
        efficiencyPct: 84,
        spinDirection: '00:42',
        gyroDegree: 33,
        activeSpinRpm: 1762,
        totalSpinRpm: 2098,
        plateXCm: 6.1,
        plateYCm: 79.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 66.2, 25.1, 6.1, 79.0),
      },
      {
        id: 'p-0826-4',
        pitchNumber: 4,
        timestamp: '2026/08/26 17:47:58',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 102.9,
        verticalBreakCm: 53.7,
        horizontalBreakCm: 25.3,
        efficiencyPct: 83,
        spinDirection: '00:50',
        gyroDegree: 33,
        activeSpinRpm: 1616,
        totalSpinRpm: 1938,
        plateXCm: -11.0,
        plateYCm: 71.5,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 53.7, 25.3, -11.0, 71.5),
      },
      {
        id: 'p-0826-5',
        pitchNumber: 5,
        timestamp: '2026/08/26 17:48:14',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 104.6,
        verticalBreakCm: 54.8,
        horizontalBreakCm: 8.8,
        efficiencyPct: 81,
        spinDirection: '00:18',
        gyroDegree: 36,
        activeSpinRpm: 1640,
        totalSpinRpm: 2036,
        plateXCm: 2.4,
        plateYCm: 84.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 54.8, 8.8, 2.4, 84.0),
      },
      {
        id: 'p-0826-6',
        pitchNumber: 6,
        timestamp: '2026/08/26 17:48:23',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 101.5,
        verticalBreakCm: 33.2,
        horizontalBreakCm: 21.0,
        efficiencyPct: 79,
        spinDirection: '01:05',
        gyroDegree: 38,
        activeSpinRpm: 1560,
        totalSpinRpm: 1975,
        plateXCm: 14.5,
        plateYCm: 64.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 33.2, 21.0, 14.5, 64.0),
      },
    ],
  },
  {
    id: 'session-2026-08-26-hit',
    title: '2026/08/26 打擊',
    date: '2026/08/26',
    monthGroup: '2026年8月',
    discipline: 'hitting',
    cameraSetup: 'behind_catcher',
    pitches: [
      {
        id: 'h-0826-1',
        pitchNumber: 1,
        timestamp: '2026/08/26 18:05:12',
        pitchType: '平飛長打 (Barrel)',
        pitchColor: '#2563eb',
        cameraAngle: 'behind_catcher',
        speedKmh: 138.4,
        verticalBreakCm: 22.5,
        horizontalBreakCm: -11.0,
        efficiencyPct: 91,
        spinDirection: '06:15',
        gyroDegree: 24,
        activeSpinRpm: 1820,
        totalSpinRpm: 2000,
        plateXCm: 0,
        plateYCm: 78,
        isHitting: true,
        launchAngleDeg: 22.5,
        sprayAngleDeg: -11.0,
        distanceM: 98.4,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 48.0, 15.0, -10, 88),
      },
      {
        id: 'h-0826-2',
        pitchNumber: 2,
        timestamp: '2026/08/26 18:06:45',
        pitchType: '外野平飛安打',
        pitchColor: '#059669',
        cameraAngle: 'behind_catcher',
        speedKmh: 131.2,
        verticalBreakCm: 16.0,
        horizontalBreakCm: 8.5,
        efficiencyPct: 88,
        spinDirection: '05:45',
        gyroDegree: 28,
        activeSpinRpm: 1650,
        totalSpinRpm: 1875,
        plateXCm: 4,
        plateYCm: 74,
        isHitting: true,
        launchAngleDeg: 16.0,
        sprayAngleDeg: 8.5,
        distanceM: 82.6,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 35.0, -12.0, 8, 76),
      },
    ],
  },
  {
    id: 'session-2026-08-19-pitch',
    title: '2026/08/19 投球',
    date: '2026/08/19',
    monthGroup: '2026年8月',
    discipline: 'pitching',
    cameraSetup: 'behind_pitcher',
    pitches: [
      {
        id: 'p-0819-1',
        pitchNumber: 1,
        timestamp: '2026/08/19 16:20:11',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 103.4,
        verticalBreakCm: 41.2,
        horizontalBreakCm: 22.4,
        efficiencyPct: 82,
        spinDirection: '00:55',
        gyroDegree: 35,
        activeSpinRpm: 1620,
        totalSpinRpm: 1975,
        plateXCm: -5.0,
        plateYCm: 77.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 41.2, 22.4, -5.0, 77.0),
      },
      {
        id: 'p-0819-2',
        pitchNumber: 2,
        timestamp: '2026/08/19 16:21:40',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 100.8,
        verticalBreakCm: 36.5,
        horizontalBreakCm: 19.8,
        efficiencyPct: 78,
        spinDirection: '00:58',
        gyroDegree: 38,
        activeSpinRpm: 1530,
        totalSpinRpm: 1960,
        plateXCm: 7.2,
        plateYCm: 70.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 36.5, 19.8, 7.2, 70.0),
      },
      {
        id: 'p-0819-3',
        pitchNumber: 3,
        timestamp: '2026/08/19 16:23:05',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_pitcher',
        speedKmh: 100.3,
        verticalBreakCm: 36.3,
        horizontalBreakCm: 19.6,
        efficiencyPct: 79,
        spinDirection: '00:56',
        gyroDegree: 37,
        activeSpinRpm: 1545,
        totalSpinRpm: 1955,
        plateXCm: 1.5,
        plateYCm: 73.5,
        ribbonPath: buildRibbonPathFromMetrics('behind_pitcher', 36.3, 19.6, 1.5, 73.5),
      },
    ],
  },
  {
    id: 'session-ai-coach-demo',
    title: 'AI教練 範例（14歲）',
    date: '2026/08/19',
    monthGroup: '2026年8月',
    discipline: 'pitching',
    cameraSetup: 'behind_catcher',
    pitches: [
      {
        id: 'p-demo-1',
        pitchNumber: 1,
        timestamp: '2026/08/19 15:10:00',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_catcher',
        speedKmh: 114.5,
        verticalBreakCm: 50.2,
        horizontalBreakCm: 19.4,
        efficiencyPct: 89,
        spinDirection: '00:42',
        gyroDegree: 27,
        activeSpinRpm: 1910,
        totalSpinRpm: 2146,
        plateXCm: -8.0,
        plateYCm: 88.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_catcher', 50.2, 19.4, -8.0, 88.0),
      },
      {
        id: 'p-demo-2',
        pitchNumber: 2,
        timestamp: '2026/08/19 15:11:15',
        pitchType: '四縫線快速球',
        pitchColor: '#be1e3c',
        cameraAngle: 'behind_catcher',
        speedKmh: 111.7,
        verticalBreakCm: 46.4,
        horizontalBreakCm: 17.0,
        efficiencyPct: 87,
        spinDirection: '00:40',
        gyroDegree: 29,
        activeSpinRpm: 1840,
        totalSpinRpm: 2115,
        plateXCm: 9.5,
        plateYCm: 81.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_catcher', 46.4, 17.0, 9.5, 81.0),
      },
      {
        id: 'p-demo-3',
        pitchNumber: 3,
        timestamp: '2026/08/19 15:12:30',
        pitchType: '滑球',
        pitchColor: '#ca8a04',
        cameraAngle: 'behind_catcher',
        speedKmh: 101.2,
        verticalBreakCm: 6.5,
        horizontalBreakCm: -24.8,
        efficiencyPct: 54,
        spinDirection: '09:30',
        gyroDegree: 57,
        activeSpinRpm: 1180,
        totalSpinRpm: 2185,
        plateXCm: -18.5,
        plateYCm: 54.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_catcher', 6.5, -24.8, -18.5, 54.0),
      },
      {
        id: 'p-demo-4',
        pitchNumber: 4,
        timestamp: '2026/08/19 15:14:00',
        pitchType: '曲球',
        pitchColor: '#0284c7',
        cameraAngle: 'behind_catcher',
        speedKmh: 94.8,
        verticalBreakCm: -34.2,
        horizontalBreakCm: -18.5,
        efficiencyPct: 76,
        spinDirection: '07:05',
        gyroDegree: 40,
        activeSpinRpm: 1690,
        totalSpinRpm: 2224,
        plateXCm: -4.0,
        plateYCm: 50.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_catcher', -34.2, -18.5, -4.0, 50.0),
      },
      {
        id: 'p-demo-5',
        pitchNumber: 5,
        timestamp: '2026/08/19 15:15:20',
        pitchType: '變速球',
        pitchColor: '#059669',
        cameraAngle: 'behind_catcher',
        speedKmh: 102.6,
        verticalBreakCm: 24.5,
        horizontalBreakCm: 31.2,
        efficiencyPct: 84,
        spinDirection: '01:45',
        gyroDegree: 33,
        activeSpinRpm: 1470,
        totalSpinRpm: 1750,
        plateXCm: 15.0,
        plateYCm: 56.0,
        ribbonPath: buildRibbonPathFromMetrics('behind_catcher', 24.5, 31.2, 15.0, 56.0),
      },
    ],
  },
];

/**
 * 將 benchmark_100.json 的 100 部投球模型轉換為 SmartScout 場次，讓使用者能在「場次」與「影片」直接檢視 100 球
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
      sdList: string[];
    }
  > = {
    four_seam: {
      name: '四縫線快速球',
      color: '#be1e3c',
      vbRange: [44, 65],
      hbRange: [12, 34],
      effRange: [82, 95],
      sdList: ['00:35', '00:48', '01:02', '00:24'],
    },
    sinker: {
      name: '二縫線伸卡球',
      color: '#ea580c',
      vbRange: [22, 39],
      hbRange: [30, 48],
      effRange: [80, 91],
      sdList: ['01:35', '01:50', '01:22', '02:05'],
    },
    slider: {
      name: '滑球',
      color: '#ca8a04',
      vbRange: [-6, 14],
      hbRange: [-36, -15],
      effRange: [46, 66],
      sdList: ['09:15', '09:45', '08:50', '10:05'],
    },
    changeup: {
      name: '變速球',
      color: '#059669',
      vbRange: [18, 34],
      hbRange: [24, 42],
      effRange: [78, 89],
      sdList: ['01:40', '01:55', '01:25', '02:00'],
    },
    curveball: {
      name: '曲球',
      color: '#0284c7',
      vbRange: [-48, -22],
      hbRange: [-28, -10],
      effRange: [72, 86],
      sdList: ['06:45', '07:15', '06:30', '07:30'],
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
    const sd = meta.sdList[idx % meta.sdList.length];
    const camAngle: 'behind_pitcher' | 'behind_catcher' =
      idx % 2 === 0 ? 'behind_pitcher' : 'behind_catcher';
    const plateX = Number(((ratio2 - 0.5) * 38).toFixed(1));
    const plateY = Number((52 + ratio * 48).toFixed(1));

    return {
      id: `bench-100-${p.id}`,
      pitchNumber: p.id,
      timestamp: `2026/11/21 18:${String(Math.floor(p.id / 2) % 60).padStart(2, '0')}:${String((p.id * 17) % 60).padStart(2, '0')}`,
      pitchType: meta.name,
      pitchColor: meta.color,
      cameraAngle: camAngle,
      speedKmh: p.predictedKmh,
      verticalBreakCm: vb,
      horizontalBreakCm: hb,
      efficiencyPct: eff,
      spinDirection: sd,
      gyroDegree: gyro,
      activeSpinRpm: activeSpin,
      totalSpinRpm: totalSpin,
      plateXCm: plateX,
      plateYCm: plateY,
      ribbonPath: buildRibbonPathFromMetrics(camAngle, vb, hb, plateX, plateY),
    };
  });

  return {
    id: 'session-benchmark-100',
    title: 'AI 100部投球模型實測庫 (MAE 0.40 km/h)',
    date: '2026/11/21',
    monthGroup: '2026年11月',
    discipline: 'pitching',
    cameraSetup: 'behind_pitcher',
    pitches,
  };
}
