import {
  BallDetection,
  CameraViewMode,
  Point2D,
  SmartScoutHitMetrics,
  SmartScoutPitchMetrics,
  SpeedResult,
  TunnelBoxConfig,
} from '../types';

export const DEFAULT_TUNNEL_CONFIGS: Record<
  'behind_pitcher' | 'behind_catcher',
  TunnelBoxConfig
> = {
  // 投手正後方視角：投手出手框在上方較大(近端)，本壘九宮格好球帶在中央偏下較小(遠端 18.44m)
  behind_pitcher: {
    releaseBox: { cx: 0.48, cy: 0.26, width: 0.28, height: 0.34 },
    strikeZoneBox: { cx: 0.50, cy: 0.58, width: 0.13, height: 0.21 },
  },
  // 捕手/本壘正後方視角(投手正前方)：遠端投手出手框在中央偏上較小，本壘九宮格好球帶在近端中央較大
  behind_catcher: {
    releaseBox: { cx: 0.50, cy: 0.34, width: 0.12, height: 0.18 },
    strikeZoneBox: { cx: 0.50, cy: 0.56, width: 0.24, height: 0.38 },
  },
};

interface PitchFlightParams {
  detections: BallDetection[];
  videoWidth: number;
  videoHeight: number;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  distanceMeters: number;
  extensionMeters?: number;
  checkpoints?: TunnelBoxConfig;
}

export const MAX_PITCH_SAMPLE_GAP_MS = 145;

/** Infer endpoints from visible held-ball, flight and stopped-ball observations. */
export function estimateAutomaticPitchFlight({
  detections, videoWidth, videoHeight, distanceMeters, extensionMeters = 1.75,
}: PitchFlightParams) {
  const distance = distanceMeters - extensionMeters;
  if (!(videoWidth > 0 && videoHeight > 0 && distance > 0 && Number.isFinite(distance))) return null;
  const points = detections.map((d) => ({ x: d.x / videoWidth, y: d.y / videoHeight }));
  const steps = detections.slice(1).map((d, i) => {
    const dt = d.timestampMs - detections[i].timestampMs;
    const x = points[i + 1].x - points[i].x;
    const y = points[i + 1].y - points[i].y;
    const valid = dt > 0 && dt <= MAX_PITCH_SAMPLE_GAP_MS &&
      [detections[i], d].every((sample) => sample.confidence >= 0.22 &&
        sample.width > 0 && sample.height > 0 && sample.x >= 0 &&
        sample.x <= videoWidth && sample.y >= 0 && sample.y <= videoHeight);
    return { x, y, dt, valid, rate: Math.hypot(x, y) / dt };
  });
  const stationary = (i: number) => steps[i]?.valid && steps[i].rate < 0.00012;
  const depthStationary = (i: number) => steps[i]?.valid && steps[i].rate < 0.000038;

  for (let start = 1; start + 6 < steps.length; start++) {
    if (!stationary(start - 1) || !steps[start].valid) continue;
    for (let end = start + 5; end + 1 < steps.length; end++) {
      if (!stationary(end) || !stationary(end + 1)) continue;
      const first = points[start];
      const last = points[end];
      const dx = last.x - first.x;
      const dy = last.y - first.y;
      const span = Math.hypot(dx, dy);
      const flightTimeMs = detections[end].timestampMs - detections[start].timestampMs;
      const speedKmh = distance * 3600 / flightTimeMs / 0.955;
      const startSize = Math.max(detections[start].width, detections[start].height);
      const endSize = Math.max(detections[end].width, detections[end].height);
      const isDepthPerspectiveArc =
        startSize > endSize * 1.15 || endSize > startSize * 1.15;

      if (isDepthPerspectiveArc) {
        if (
          !depthStationary(start - 1) ||
          depthStationary(start) ||
          !depthStationary(end) ||
          !depthStationary(end + 1)
        ) continue;
        if (
          span < 0.014 ||
          speedKmh < 40 ||
          speedKmh > 158 ||
          [first, last].some(({ x, y }) => x < 0.04 || x > 0.96 || y < 0.04 || y > 0.96)
        ) continue;
        let hasValidFlight = true;
        for (let k = start; k < end; k++) {
          if (!steps[k].valid || (k + 1 < end && depthStationary(k) && depthStationary(k + 1))) {
            hasValidFlight = false;
            break;
          }
        }
        if (!hasValidFlight) continue;
        const track = detections.slice(start, end + 1);
        return { track, flightTimeMs, speedKmh: Number(speedKmh.toFixed(1)), detectionCount: track.length };
      }

      if (stationary(start)) continue;
      if (span < 0.065 || speedKmh < 40 || speedKmh > 158 ||
          [first, last].some(({ x, y }) => x < 0.04 || x > 0.96 || y < 0.04 || y > 0.96)) continue;

      const flight = steps.slice(start, end);
      const rates = flight.map((step) => step.rate).sort((a, b) => a - b);
      const median = rates[Math.floor(rates.length / 2)];
      const advances = flight.filter((step) => step.x * dx + step.y * dy > 0).length;
      if (advances / flight.length < 0.8 || flight.some((step) =>
        !step.valid || step.rate > median * 4 || (step.x * dx + step.y * dy) / span < -0.012
      ) || steps[start - 1].rate > median * 0.25 ||
        Math.max(steps[end].rate, steps[end + 1].rate) > median * 0.25) continue;

      const track = detections.slice(start, end + 1);
      return { track, flightTimeMs, speedKmh: Number(speedKmh.toFixed(1)), detectionCount: track.length };
    }
  }

  // Segment-aware multi-angle & frame-drop resilient fallback:
  // Splits detections into contiguous valid segments (handles multi-pitch clips) and
  // supports both 3D depth-perspective angles (behind_pitcher / behind_catcher, sizeRatio >= 1.35)
  // and oblique/side angles (large 2D lateral sweep span >= 0.14).
  if (detections.length >= 6) {
    let segStart = 0;
    for (let i = 0; i <= steps.length; i++) {
      const isBreak = i === steps.length || !steps[i]?.valid;
      if (!isBreak) continue;
      const segEnd = i;
      if (segEnd - segStart >= 5) {
        let s = segStart;
        let e = segEnd;
        while (s < e && depthStationary(s)) s++;
        while (e > s && depthStationary(e - 1)) e--;
        if (e - s >= 5) {
          const firstDet = detections[s];
          const lastDet = detections[e];
          const startSize = Math.max(firstDet.width, firstDet.height);
          const endSize = Math.max(lastDet.width, lastDet.height);
          const sizeRatio = Math.max(startSize / Math.max(1, endSize), endSize / Math.max(1, startSize));
          const first = points[s];
          const last = points[e];
          const span = Math.hypot(last.x - first.x, last.y - first.y);
          const hasEndStop = e < segEnd || sizeRatio >= 1.45;
          const isDepthTrack = sizeRatio >= 1.35 && span >= 0.014;
          const isObliqueTrack = sizeRatio >= 1.15 && span >= 0.14 && e < segEnd;
          const releasePadMs = s === segStart && steps[s]?.dt > 45 ? 33.37 : 0;
          const catchPadMs =
            e === segEnd && steps[e - 1]?.dt > 45 && sizeRatio >= 1.50 ? 33.37 : 0;
          const flightTimeMs = lastDet.timestampMs - firstDet.timestampMs - releasePadMs - catchPadMs;
          const speedKmh = (distance * 3600) / flightTimeMs / 0.955;
          if (
            (isDepthTrack || isObliqueTrack) &&
            hasEndStop &&
            flightTimeMs >= 320 &&
            flightTimeMs <= 920 &&
            speedKmh >= 45 &&
            speedKmh <= 158 &&
            ![first, last].some(({ x, y }) => x < 0.04 || x > 0.96 || y < 0.04 || y > 0.96)
          ) {
            const track = detections.slice(s, e + 1);
            return {
              track,
              flightTimeMs: Number(flightTimeMs.toFixed(1)),
              speedKmh: Number(speedKmh.toFixed(1)),
              detectionCount: track.length,
            };
          }
        }
      }
      segStart = i + 1;
    }
  }
  return null;
}

/**
 * Estimate the release speed from a track that covers both image checkpoints.
 * Image checkpoints and the default extension are approximations. This method
 * does not infer 3D positions, spin or strike location from the boxes.
 */
export function estimatePitchFlight({
  detections,
  videoWidth,
  videoHeight,
  cameraAngle,
  distanceMeters,
  extensionMeters = 1.75,
  checkpoints = DEFAULT_TUNNEL_CONFIGS[cameraAngle],
}: PitchFlightParams) {
  const flightDistanceMeters = distanceMeters - extensionMeters;
  if (
    videoWidth <= 0 ||
    videoHeight <= 0 ||
    !Number.isFinite(flightDistanceMeters) ||
    flightDistanceMeters <= 0
  ) return null;

  const inBox = (sample: BallDetection, box: TunnelBoxConfig['releaseBox']) =>
    Math.abs(sample.x / videoWidth - box.cx) <= box.width / 2 &&
    Math.abs(sample.y / videoHeight - box.cy) <= box.height / 2;

  const directionX = checkpoints.strikeZoneBox.cx - checkpoints.releaseBox.cx;
  const directionY = checkpoints.strikeZoneBox.cy - checkpoints.releaseBox.cy;
  const directionLength = Math.hypot(directionX, directionY);
  if (directionLength < 0.05) return null;
  const alongFlight = (sample: BallDetection) =>
    ((sample.x / videoWidth - checkpoints.releaseBox.cx) * directionX +
      (sample.y / videoHeight - checkpoints.releaseBox.cy) * directionY) /
    (directionLength * directionLength);

  let track: BallDetection[] = [];
  let priorPosition = 0;
  for (const detection of detections) {
    if (
      !Number.isFinite(detection.timestampMs) ||
      !(detection.confidence > 0 && detection.confidence <= 1) ||
      !(detection.width > 0 && detection.height > 0) ||
      !(detection.x >= 0 && detection.x <= videoWidth) ||
      !(detection.y >= 0 && detection.y <= videoHeight)
    ) continue;

    const position = alongFlight(detection);
    if (track.length) {
      const previous = track[track.length - 1];
      const gapMs = detection.timestampMs - previous.timestampMs;
      if (gapMs <= 0) continue;
      if (
        gapMs > MAX_PITCH_SAMPLE_GAP_MS ||
        Math.abs(position - priorPosition) > 0.55 ||
        position < priorPosition - 0.2 ||
        detection.timestampMs - track[0].timestampMs > 1600
      ) {
        track = [];
      } else if (
        track.length === 1 &&
        position <= 0.25 &&
        inBox(detection, checkpoints.releaseBox) &&
        Math.hypot(
          (detection.x - previous.x) / videoWidth,
          (detection.y - previous.y) / videoHeight,
        ) < 0.006
      ) {
        // A ball held still before the pitch does not start the flight clock.
        track = [];
      }
    }

    if (!track.length && (!inBox(detection, checkpoints.releaseBox) || position > 0.25)) {
      continue;
    }

    const previousSample = track[track.length - 1];
    const previousPosition = priorPosition;
    track.push(detection);
    priorPosition = position;
    if (
      track.length < 5 ||
      position < 0.98 ||
      !inBox(detection, checkpoints.strikeZoneBox)
    ) continue;

    const plateTimestampMs = previousPosition < 1 && position > 1
      ? previousSample.timestampMs +
        ((1 - previousPosition) / (position - previousPosition)) *
          (detection.timestampMs - previousSample.timestampMs)
      : detection.timestampMs;
    const flightTimeMs = plateTimestampMs - track[0].timestampMs;
    const speedKmh = (flightDistanceMeters / (flightTimeMs / 1000)) * 3.6 / 0.955;
    if (!Number.isFinite(speedKmh) || speedKmh < 35 || speedKmh > 195) continue;

    return {
      speedKmh: Number(speedKmh.toFixed(1)),
      flightTimeMs,
      detectionCount: track.length,
      track,
    };
  }
  return null;
}

export const PITCH_COLOR_MAP: Record<string, { name: string; short: string; color: string }> = {
  four_seam: { name: '四縫線速球 (4-Seam)', short: '四縫線', color: '#ef4444' },
  sinker: { name: '二縫線伸卡 (Sinker)', short: '伸卡球', color: '#f97316' },
  slider: { name: '高速滑球 (Slider)', short: '滑球', color: '#eab308' },
  curveball: { name: '12-6 曲球 (Curveball)', short: '曲球', color: '#38bdf8' },
  changeup: { name: '圈指變速球 (Changeup)', short: '變速球', color: '#10b981' },
  cutter: { name: '卡特球 (Cutter)', short: '卡特球', color: '#a855f7' },
};

// 標準好球帶物理尺寸：本壘板寬 43.18cm (±21.6cm)，好球帶高度 46cm ~ 106cm (中心 76cm, 半高 30cm)
const SZ_HALF_WIDTH_CM = 21.6;
const SZ_CENTER_Y_CM = 76.0;
const SZ_HALF_HEIGHT_CM = 30.0;

export function computeStrikeZoneGrid(plateXCm: number, plateYCm: number): {
  zoneId: number;
  isStrike: boolean;
  zoneLabel: string;
} {
  const normX = plateXCm / SZ_HALF_WIDTH_CM; // -1 ~ +1 為好球帶寬度
  const normY = (plateYCm - SZ_CENTER_Y_CM) / SZ_HALF_HEIGHT_CM; // -1(下緣) ~ +1(上緣)

  if (Math.abs(normX) <= 1.0 && Math.abs(normY) <= 1.0) {
    // 3x3 九宮格 (1~3 上排左至右, 4~6 中排左至右, 7~9 下排左至右)
    const col = normX < -0.333 ? 0 : normX <= 0.333 ? 1 : 2;
    const row = normY > 0.333 ? 0 : normY >= -0.333 ? 1 : 2;
    const zoneId = row * 3 + col + 1;
    return {
      zoneId,
      isStrike: true,
      zoneLabel: `好球帶 #${zoneId}`,
    };
  }

  // 外圍追打/壞球帶 11(左上), 12(右上), 13(左下), 14(右下)
  if (normY >= 0) {
    const zoneId = normX < 0 ? 11 : 12;
    return { zoneId, isStrike: false, zoneLabel: zoneId === 11 ? '壞球 B11 (高左)' : '壞球 B12 (高右)' };
  } else {
    const zoneId = normX < 0 ? 13 : 14;
    return { zoneId, isStrike: false, zoneLabel: zoneId === 13 ? '壞球 B13 (低左)' : '壞球 B14 (低右)' };
  }
}

export function degToClockString(deg: number): string {
  // 定義：+90° (純上竄背旋) = 12:00，-90° (純下墜正旋) = 6:00，0° (純向右/手臂側) = 3:00，180° (純向左/手套側) = 9:00
  const normalized = ((90 - deg) % 360 + 360) % 360;
  const totalMinutes = Math.round((normalized / 360) * 720) % 720;
  let hours = Math.floor(totalMinutes / 60);
  const mins = Math.round((totalMinutes % 60) / 15) * 15;
  if (mins === 60) {
    hours = (hours + 1) % 12;
  }
  const finalHours = hours === 0 ? 12 : hours;
  const finalMins = mins === 60 ? 0 : mins;
  return `${String(finalHours).padStart(2, '0')}:${String(finalMins).padStart(2, '0')}`;
}

export function classifyPitchFromMetrics(
  speedKmh: number,
  hbCm: number,
  ivbCm: number
): string {
  if (ivbCm < -18) return 'curveball';
  if (hbCm < -10 && ivbCm < 16) return 'slider';
  if (speedKmh >= 138 && ivbCm >= 26 && Math.abs(hbCm) <= 22) return 'four_seam';
  if (speedKmh >= 134 && hbCm > 18 && ivbCm < 26) return 'sinker';
  if (speedKmh < 137 && hbCm > 12) return 'changeup';
  if (hbCm < -4 && speedKmh >= 135) return 'cutter';
  return speedKmh >= 138 ? 'four_seam' : 'changeup';
}

export function generateAiScoutComment(
  speedKmh: number,
  pitchShort: string,
  hbCm: number,
  ivbCm: number,
  spinRateRpm: number,
  spinEfficiencyPct: number,
  zoneId: number,
  isStrike: boolean
): string {
  const locText = isStrike
    ? zoneId === 5
      ? '進壘位置偏紅中 (#5)，需注意被打者鎖定'
      : [1, 3, 7, 9].includes(zoneId)
        ? `精準壓在九宮格邊角 (#${zoneId})，具備極佳壓迫感`
        : `穩定進入好球帶 (#${zoneId})`
    : `落在好球帶外圍追打區 (B${zoneId})，可作為引誘球`;

  if (ivbCm >= 35) {
    return `🔥 頂尖上竄尾勁！${pitchShort}初速 ${speedKmh.toFixed(1)} km/h 搭配 +${ivbCm.toFixed(1)}cm 垂直上竄與 ${spinRateRpm} RPM 高轉速（效率 ${spinEfficiencyPct}%），${locText}。`;
  }
  if (ivbCm <= -22) {
    return `🌊 大幅度縱向下墜！${pitchShort}垂直落差達 ${ivbCm.toFixed(1)}cm（轉速 ${spinRateRpm} RPM），能有效破壞打者擊球視角，${locText}。`;
  }
  if (Math.abs(hbCm) >= 22) {
    const dir = hbCm > 0 ? '手臂側橫移 (Arm-Side Run)' : '手套側橫向滑移 (Glove-Side Sweep)';
    return `⚡ 犀利橫向位移！${pitchShort}展現 ${hbCm > 0 ? '+' : ''}${hbCm.toFixed(1)}cm ${dir}，${locText}。`;
  }
  return `⚾ ${pitchShort}初速 ${speedKmh.toFixed(1)} km/h（轉速 ${spinRateRpm} RPM，效率 ${spinEfficiencyPct}%，位移 HB ${hbCm >= 0 ? '+' : ''}${hbCm.toFixed(1)}cm / IVB ${ivbCm >= 0 ? '+' : ''}${ivbCm.toFixed(1)}cm），${locText}。`;
}

export function buildSmartScoutMetrics(params: {
  speedKmh: number;
  pitchType?: string;
  horizontalBreakCm: number;
  verticalBreakCm: number;
  plateXCm: number;
  plateYCm: number;
  spinRateRpm?: number;
  spinEfficiencyPct?: number;
  releaseHeightM?: number;
  releaseSideM?: number;
  extensionM?: number;
}): SmartScoutPitchMetrics {
  const {
    speedKmh,
    horizontalBreakCm: hbCm,
    verticalBreakCm: ivbCm,
    plateXCm,
    plateYCm,
  } = params;

  const pitchType =
    params.pitchType && PITCH_COLOR_MAP[params.pitchType]
      ? params.pitchType
      : classifyPitchFromMetrics(speedKmh, hbCm, ivbCm);
  const meta = PITCH_COLOR_MAP[pitchType] ?? PITCH_COLOR_MAP.four_seam;

  // 由位移向量計算旋轉軸角度與有效轉速
  const breakMagCm = Math.hypot(hbCm, ivbCm);
  const spinAxisDeg = Math.round((Math.atan2(ivbCm, hbCm) * 180) / Math.PI);
  const spinAxisClock = degToClockString(spinAxisDeg);

  const defaultEff =
    pitchType === 'four_seam'
      ? 94
      : pitchType === 'sinker'
        ? 89
        : pitchType === 'changeup'
          ? 86
          : pitchType === 'curveball'
            ? 79
            : 58; // 滑球/卡特球具較高陀螺旋轉分量

  const spinEfficiencyPct = Math.max(
    32,
    Math.min(99, Math.round(params.spinEfficiencyPct ?? defaultEff))
  );

  // 依據空氣動力馬格努斯位移反推有效轉速與總轉速
  const estActiveSpin = Math.round(
    Math.min(2750, Math.max(950, breakMagCm * 36 + speedKmh * 5.2))
  );
  const spinRateRpm = Math.round(
    params.spinRateRpm ?? Math.min(3150, Math.max(1450, estActiveSpin / (spinEfficiencyPct / 100)))
  );
  const activeSpinRpm = Math.round(spinRateRpm * (spinEfficiencyPct / 100));
  const gyroAngleDeg = Number(
    ((Math.acos(spinEfficiencyPct / 100) * 180) / Math.PI).toFixed(1)
  );

  const plateSpeedKmh = Number((speedKmh * 0.916).toFixed(1));
  const { zoneId, isStrike } = computeStrikeZoneGrid(plateXCm, plateYCm);

  const aiScoutComment = generateAiScoutComment(
    speedKmh,
    meta.short,
    hbCm,
    ivbCm,
    spinRateRpm,
    spinEfficiencyPct,
    zoneId,
    isStrike
  );

  return {
    pitchType,
    pitchName: meta.name,
    pitchShort: meta.short,
    pitchColor: meta.color,
    plateSpeedKmh,
    spinRateRpm,
    activeSpinRpm,
    spinEfficiencyPct,
    spinAxisClock,
    spinAxisDeg,
    gyroAngleDeg,
    verticalBreakCm: Number(ivbCm.toFixed(1)),
    horizontalBreakCm: Number(hbCm.toFixed(1)),
    releaseHeightM: Number((params.releaseHeightM ?? 1.74).toFixed(2)),
    releaseSideM: Number((params.releaseSideM ?? -0.42).toFixed(2)),
    extensionM: Number((params.extensionM ?? 1.88).toFixed(2)),
    plateXCm: Number(plateXCm.toFixed(1)),
    plateYCm: Number(plateYCm.toFixed(1)),
    zoneId,
    isStrike,
    aiScoutComment,
  };
}

/**
 * 將 3D 投球通道軌跡投影為「投手正後方」或「捕手/本壘正後方」2D 畫面像素偵測點
 */
export function project3DTunnelPitchToDetections(params: {
  speedKmh: number;
  horizontalBreakCm: number;
  verticalBreakCm: number;
  plateXCm: number;
  plateYCm: number;
  releaseHeightM?: number;
  releaseSideM?: number;
  fps: number;
  viewMode: 'behind_pitcher' | 'behind_catcher';
  tunnelConfig: TunnelBoxConfig;
  videoWidth: number;
  videoHeight: number;
}): BallDetection[] {
  const {
    speedKmh,
    horizontalBreakCm,
    verticalBreakCm,
    plateXCm,
    plateYCm,
    fps,
    viewMode,
    tunnelConfig,
    videoWidth,
    videoHeight,
  } = params;

  const distanceM = 18.44;
  const zRelease = 1.75;
  const flightDistM = distanceM - zRelease;
  const avgSpeedMps = (speedKmh * 0.955) / 3.6;
  const totalTimeS = flightDistM / avgSpeedMps;
  const numFrames = Math.max(8, Math.round(totalTimeS * fps));

  const relBox = tunnelConfig.releaseBox;
  const szBox = tunnelConfig.strikeZoneBox;

  // 出手點在出手框內的相對偏移
  const relStartX = (relBox.cx + (params.releaseSideM ?? -0.35) * relBox.width * 0.35) * videoWidth;
  const relStartY =
    (relBox.cy - ((params.releaseHeightM ?? 1.75) - 1.55) * relBox.height * 0.45) * videoHeight;

  // 進壘點在九宮格好球帶框內的真實對應像素座標
  const normPlateX = plateXCm / SZ_HALF_WIDTH_CM; // -1 ~ +1 為九宮格左右邊界
  const normPlateY = (plateYCm - SZ_CENTER_Y_CM) / SZ_HALF_HEIGHT_CM; // -1(下) ~ +1(上)
  const targetPixelX = (szBox.cx + normPlateX * (szBox.width * 0.5)) * videoWidth;
  const targetPixelY = (szBox.cy - normPlateY * (szBox.height * 0.5)) * videoHeight;

  // 位移弧線像素幅度 (中段彎曲與重力拋物線拱高)
  const pxPerCmAtPlate = (szBox.width * videoWidth) / (SZ_HALF_WIDTH_CM * 2);
  const curveOffsetX = horizontalBreakCm * pxPerCmAtPlate * 0.55;
  const gravityArchY = -42 * pxPerCmAtPlate - verticalBreakCm * pxPerCmAtPlate * 0.48;

  // 近端與遠端球體視覺直徑 (像素)
  const nearDiameterPx = viewMode === 'behind_pitcher' ? 32 : 34;
  const farDiameterPx = viewMode === 'behind_pitcher' ? 11 : 10;

  const detections: BallDetection[] = [];

  for (let i = 0; i <= numFrames; i++) {
    const u = i / numFrames; // 時間進度 0 -> 1

    // 透視深度權重：近端移動快、遠端移動慢 (針孔相機 1/Z 投影定律)
    const perspectiveS =
      viewMode === 'behind_pitcher'
        ? (u * 2.4) / (1 + u * 1.4) // 投手後方：剛出手時在近端視覺位移快，接近本壘時收斂
        : (u * 0.42) / (1 - u * 0.58); // 捕手後方：遠端出手時視覺位移慢，接近本壘時急遽放大加速

    const archWeight = 4 * u * (1 - u);
    const x =
      relStartX +
      (targetPixelX - relStartX) * perspectiveS -
      curveOffsetX * archWeight;
    const y =
      relStartY +
      (targetPixelY - relStartY) * perspectiveS +
      gravityArchY * archWeight;

    const diam =
      viewMode === 'behind_pitcher'
        ? nearDiameterPx + (farDiameterPx - nearDiameterPx) * perspectiveS
        : farDiameterPx + (nearDiameterPx - farDiameterPx) * perspectiveS;

    detections.push({
      x: Number(x.toFixed(1)),
      y: Number(y.toFixed(1)),
      width: Number(diam.toFixed(1)),
      height: Number(diam.toFixed(1)),
      confidence: Number((0.86 + 0.1 * Math.sin(u * Math.PI)).toFixed(2)),
      timestampMs: Number(((u * totalTimeS) * 1000).toFixed(2)),
    });
  }

  return detections;
}

/**
 * SmartScout 縱深雙框追蹤引擎：支援直接從「投手正後方」或「捕手/本壘正後方」即時推算球速、九宮格進壘點、位移與轉速
 */
export class SmartScoutTunnelEstimator {
  private viewMode: 'behind_pitcher' | 'behind_catcher';
  private tunnelConfig: TunnelBoxConfig;
  private videoWidth: number;
  private videoHeight: number;
  private distanceMeters: number;
  private samples: BallDetection[] = [];

  constructor(
    viewMode: 'behind_pitcher' | 'behind_catcher',
    tunnelConfig: TunnelBoxConfig,
    videoWidth = 1280,
    videoHeight = 720,
    distanceMeters = 18.44
  ) {
    this.viewMode = viewMode;
    this.tunnelConfig = tunnelConfig;
    this.videoWidth = videoWidth;
    this.videoHeight = videoHeight;
    this.distanceMeters = distanceMeters;
  }

  public reset(): void {
    this.samples = [];
  }

  public pushDetection(det: BallDetection): SpeedResult | null {
    if (det.confidence < 0.28) return null;

    if (this.samples.length > 0) {
      const prev = this.samples[this.samples.length - 1];
      const dt = det.timestampMs - prev.timestampMs;
      if (dt <= 0) return null;
      if (dt > 240) {
        const finalized = this.computeResult();
        this.samples = [det];
        return finalized;
      }
    }

    this.samples.push(det);
    if (this.samples.length >= 5) {
      return this.computeResult();
    }
    return null;
  }

  public flush(): SpeedResult | null {
    const res = this.computeResult();
    this.samples = [];
    return res;
  }

  private computeResult(): SpeedResult | null {
    const n = this.samples.length;
    if (n < 5) return null;

    const first = this.samples[0];
    const last = this.samples[n - 1];
    const flightTimeMs = last.timestampMs - first.timestampMs;
    if (flightTimeMs < 180 || flightTimeMs > 1200) return null;

    // 1. 由飛行時間與投手延伸距離 (18.44m - 1.75m = 16.69m) + 空氣阻力補償係數反推出手初速
    const effectiveFlightM = Math.max(8, this.distanceMeters - 1.75);
    const avgSpeedMps = effectiveFlightM / (flightTimeMs / 1000);
    const releaseSpeedKmh = Number((avgSpeedMps * 3.6 / 0.955).toFixed(1));
    if (releaseSpeedKmh < 35 || releaseSpeedKmh > 195) return null;

    // 2. 由最終進壘幀在九宮格好球帶框內的相對位置反推本壘進壘座標 (plateXCm, plateYCm)
    const szBox = this.tunnelConfig.strikeZoneBox;
    const szCenterX = szBox.cx * this.videoWidth;
    const szCenterY = szBox.cy * this.videoHeight;
    const szHalfW = Math.max(12, szBox.width * this.videoWidth * 0.5);
    const szHalfH = Math.max(16, szBox.height * this.videoHeight * 0.5);

    const normPlateX = Math.max(-2.2, Math.min(2.2, (last.x - szCenterX) / szHalfW));
    const normPlateY = Math.max(-2.2, Math.min(2.2, (szCenterY - last.y) / szHalfH));

    const plateXCm = Number((normPlateX * SZ_HALF_WIDTH_CM).toFixed(1));
    const plateYCm = Number((SZ_CENTER_Y_CM + normPlateY * SZ_HALF_HEIGHT_CM).toFixed(1));

    // 3. 由中段軌跡偏離直線之弧度反推水平位移 (HB) 與垂直位移 (IVB)
    const mid = this.samples[Math.floor(n / 2)];
    const lineMidX = (first.x + last.x) * 0.5;
    const lineMidY = (first.y + last.y) * 0.5;
    const pxPerCm = szHalfW / SZ_HALF_WIDTH_CM;

    const rawDeflectXCm = (lineMidX - mid.x) / Math.max(0.5, pxPerCm * 0.55);
    const rawDeflectYCm = ((lineMidY - mid.y) / Math.max(0.5, pxPerCm) - 42) / 0.48;

    const horizontalBreakCm = Number(Math.max(-52, Math.min(52, rawDeflectXCm)).toFixed(1));
    const verticalBreakCm = Number(Math.max(-52, Math.min(52, rawDeflectYCm)).toFixed(1));

    const smartScout = buildSmartScoutMetrics({
      speedKmh: releaseSpeedKmh,
      horizontalBreakCm,
      verticalBreakCm,
      plateXCm,
      plateYCm,
    });

    const avgConf = this.samples.reduce((acc, s) => acc + s.confidence, 0) / n;
    const confidenceScore = Number(Math.min(0.99, 0.72 + avgConf * 0.25).toFixed(3));

    return {
      speedKmh: releaseSpeedKmh,
      speedMph: Number((releaseSpeedKmh * 0.621371).toFixed(1)),
      flightTimeMs: Math.round(flightTimeMs),
      confidenceScore,
      viewMode: this.viewMode,
      trajectoryWorld: this.samples.map((s, idx) => ({
        x: Number(((idx / Math.max(1, n - 1)) * this.distanceMeters).toFixed(2)),
        y: Number((1.75 - (idx / Math.max(1, n - 1)) * 0.95).toFixed(2)),
      })),
      trajectoryPixel: this.samples.map((s) => ({
        x: s.x,
        y: s.y,
        r: Math.max(4, Math.min(22, s.width * 0.45)),
      })),
      smartScout,
    };
  }
}

/**
 * SmartScout 打擊分析模擬與計算器 (Exit Velocity / Launch Angle / Spray Angle / Distance)
 */
export function generateSmartScoutHitSample(id: number, preset?: 'barrel' | 'linedrive' | 'grounder' | 'flyball'): SmartScoutHitMetrics {
  const mode =
    preset ??
    (['barrel', 'linedrive', 'flyball', 'grounder'] as const)[id % 4];

  let evKmh = 152;
  let laDeg = 24;
  let sprayDeg = -12;

  if (mode === 'barrel') {
    evKmh = Number((158 + (Math.random() * 16 - 4)).toFixed(1));
    laDeg = Number((24 + (Math.random() * 8 - 4)).toFixed(1));
    sprayDeg = Number((Math.random() * 44 - 22).toFixed(1));
  } else if (mode === 'linedrive') {
    evKmh = Number((144 + (Math.random() * 14 - 5)).toFixed(1));
    laDeg = Number((14 + (Math.random() * 8 - 4)).toFixed(1));
    sprayDeg = Number((Math.random() * 60 - 30).toFixed(1));
  } else if (mode === 'flyball') {
    evKmh = Number((138 + (Math.random() * 14 - 6)).toFixed(1));
    laDeg = Number((33 + (Math.random() * 12 - 4)).toFixed(1));
    sprayDeg = Number((Math.random() * 54 - 27).toFixed(1));
  } else {
    evKmh = Number((136 + (Math.random() * 18 - 6)).toFixed(1));
    laDeg = Number((-6 + Math.random() * 12).toFixed(1));
    sprayDeg = Number((Math.random() * 64 - 32).toFixed(1));
  }

  const evMph = Number((evKmh * 0.621371).toFixed(1));
  const vMps = evKmh / 3.6;
  const rad = (Math.max(2, laDeg) * Math.PI) / 180;
  // 考慮空氣阻力折減之飛行距離近似公式
  const rawCarry = ((vMps * vMps * Math.sin(2 * rad)) / 9.81) * 0.62;
  const estimatedDistanceM = Number(
    (laDeg < 3 ? Math.max(12, 28 + laDeg * 2.2) : Math.max(18, rawCarry)).toFixed(1)
  );

  const hitType: SmartScoutHitMetrics['hitType'] =
    laDeg < 10
      ? '滾地球 (Ground Ball)'
      : laDeg < 25
        ? '平飛球 (Line Drive)'
        : laDeg < 50
          ? '高飛球 (Fly Ball)'
          : '沖天炮 (Pop Up)';

  const isHardHit = evMph >= 95;
  const isBarrel = evMph >= 98 && laDeg >= 22 && laDeg <= 32;

  const hitResultTag =
    estimatedDistanceM >= 118 && laDeg >= 20 && laDeg <= 38
      ? '🎆 全壘打 (HOME RUN)'
      : isBarrel
        ? '🔥 完美強勁長打 (BARREL)'
        : hitType === '平飛球 (Line Drive)'
          ? '⚡ 外野安打落點 (LINE DRIVE HIT)'
          : hitType === '滾地球 (Ground Ball)'
            ? '⚾ 內野滾地球'
            : '🎯 外野飛球';

  const fieldDir =
    sprayDeg < -12 ? '左外野方向' : sprayDeg > 12 ? '右外野方向' : '中外野方向';

  const aiHitComment = isBarrel
    ? `🔥 完美擊球甜蜜點 (Barrel)！擊球初速高達 ${evKmh} km/h (${evMph} mph)、仰角 ${laDeg}°，朝${fieldDir}飛行達 ${estimatedDistanceM}m！`
    : `🏏 擊球初速 ${evKmh} km/h (${evMph} mph)、仰角 ${laDeg}°（${hitType}），朝${fieldDir}預估飛行 ${estimatedDistanceM}m。${
        laDeg < 8
          ? '建議微調揮棒軌跡提升擊球仰角至 12°~25° 平飛球帶。'
          : laDeg > 36
            ? '擊球仰角偏高，建議鎖定球心上半部壓低平飛仰角。'
            : '擊球仰角處於理想安打區間！'
      }`;

  return {
    id,
    exitVelocityKmh: evKmh,
    exitVelocityMph: evMph,
    launchAngleDeg: laDeg,
    sprayAngleDeg: sprayDeg,
    estimatedDistanceM,
    hitType,
    hitResultTag,
    isHardHit,
    isBarrel,
    aiHitComment,
  };
}
