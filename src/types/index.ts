export interface Point2D {
  x: number;
  y: number;
}

export type CameraViewMode = 'behind_pitcher' | 'behind_catcher' | 'side_view';
export type AnalysisDiscipline = 'pitching' | 'hitting';

export interface CalibrationConfig {
  distanceMeters: number;        // 預設 18.44m (投手板至本壘板)
  referenceHeightMeters: number; // 垂直切面參考高度，預設 1.5m
  // 畫面像素點：[0]投手側頂點, [1]本壘側頂點, [2]本壘側底點, [3]投手側底點
  imagePoints: [Point2D, Point2D, Point2D, Point2D];
}

export interface TunnelBoxConfig {
  // 遠端框與近端框 (0~1 正規化座標): { cx, cy, width, height }
  releaseBox: { cx: number; cy: number; width: number; height: number };
  strikeZoneBox: { cx: number; cy: number; width: number; height: number };
}

export interface BallDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  timestampMs: number;
}

export interface SmartScoutPitchMetrics {
  pitchType: string;
  pitchName: string;
  pitchShort: string;
  pitchColor: string;
  plateSpeedKmh: number;
  spinRateRpm: number;
  activeSpinRpm: number;
  spinEfficiencyPct: number;
  spinAxisClock: string;
  spinAxisDeg: number;
  gyroAngleDeg: number;
  verticalBreakCm: number;   // IVB (+上竄 / -下墜)
  horizontalBreakCm: number; // HB (+手臂側 / -手套側)
  releaseHeightM: number;
  releaseSideM: number;
  extensionM: number;
  plateXCm: number;          // 本壘水平位置 (-45 ~ +45 cm, 0 為紅中)
  plateYCm: number;          // 本壘垂直高度 (15 ~ 135 cm, 好球帶約 46 ~ 106 cm)
  zoneId: number;            // 1~9 為九宮格好球帶, 11~14 為外圍追打壞球帶
  isStrike: boolean;
  aiScoutComment: string;
}

export interface SmartScoutHitMetrics {
  id: number;
  exitVelocityKmh: number;
  exitVelocityMph: number;
  launchAngleDeg: number;
  sprayAngleDeg: number;     // -45°(左外野線) ~ +45°(右外野線)
  estimatedDistanceM: number;
  hitType: '滾地球 (Ground Ball)' | '平飛球 (Line Drive)' | '高飛球 (Fly Ball)' | '沖天炮 (Pop Up)';
  hitResultTag: string;
  isHardHit: boolean;
  isBarrel: boolean;
  aiHitComment: string;
}

export interface SpeedResult {
  speedKmh: number;
  speedMph: number;
  flightTimeMs: number;
  confidenceScore: number;
  trajectoryWorld: Point2D[];
  trajectoryPixel?: Array<{ x: number; y: number; r: number }>;
  label?: string;
  groundTruthKmh?: number;
  viewMode?: CameraViewMode;
  smartScout?: SmartScoutPitchMetrics;
}

export interface BenchmarkPitchItem {
  id: number;
  pitchType: string;
  pitchName: string;
  pitchShort: string;
  cameraProfile: string;
  normPoints: [Point2D, Point2D, Point2D, Point2D];
  fps: number;
  groundTruthKmh: number;
  groundTruthMph: number;
  predictedKmh: number;
  predictedMph: number;
  errorKmh: number;
  absErrorKmh: number;
  confidenceScore: number;
  rSquared: number;
  detectedFrames: number;
  flightTimeMs: number;
  detections: BallDetection[];
  smartScout?: SmartScoutPitchMetrics;
}

export interface BenchmarkDataset {
  generatedAt: string;
  totalPitches: number;
  modelFile: string;
  onnxProvider: string;
  avgOnnxInferMs: number;
  summary: {
    maeKmh: number;
    rmseKmh: number;
    within1_5KmhPct: number;
    within3_0KmhPct: number;
    byFps: Record<string, { count: number; maeKmh: number; maxErrKmh: number }>;
    byPitchType: Array<{
      id: string;
      name: string;
      short: string;
      count: number;
      maeKmh: number;
    }>;
  };
  pitches: BenchmarkPitchItem[];
}

export type PitchActionPhase = 'idle' | 'pitching_start' | 'pitching_end';

export type WorkerInMessage =
  | {
      type: 'INIT';
      modelUrl: string;
      actionModelUrl?: string;
      poseModelUrl?: string;
      wasmPaths: string;
      inputSize?: number;
      confThreshold?: number;
    }
  | {
      type: 'INFER';
      bitmap: ImageBitmap;
      timestampMs: number;
      origWidth: number;
      origHeight: number;
      detectAction?: boolean;
      detectPose?: boolean;
      /** 球體飛行中且全畫面漏抓遠端小球時，以此預測中心進行 2.8x 局部放大追焦 */
      predictedRoi?: { x: number; y: number; radius: number };
    };

export type WorkerOutMessage =
  | { type: 'READY'; backend: string; actionModelReady?: boolean }
  | {
      type: 'RESULT';
      detection: BallDetection | null;
      inferTimeMs: number;
      actionPhase?: PitchActionPhase;
      actionConfidence?: number;
      /** 投手 17 點骨架（像素座標），undefined = 本幀未做骨架，null = 看不到投手 */
      pose?: Array<[number, number, number]> | null;
      timestampMs?: number;
    }
  | { type: 'ERROR'; message: string };
