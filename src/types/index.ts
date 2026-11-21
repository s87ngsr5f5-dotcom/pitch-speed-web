export interface Point2D {
  x: number;
  y: number;
}

export interface CalibrationConfig {
  distanceMeters: number;        // 預設 18.44m (投手板至本壘板)
  referenceHeightMeters: number; // 垂直切面參考高度，預設 1.5m
  // 畫面像素點：[0]投手側頂點, [1]本壘側頂點, [2]本壘側底點, [3]投手側底點
  imagePoints: [Point2D, Point2D, Point2D, Point2D];
}

export interface BallDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  timestampMs: number;
}

export interface SpeedResult {
  speedKmh: number;
  speedMph: number;
  flightTimeMs: number;
  confidenceScore: number;
  trajectoryWorld: Point2D[];
  label?: string;
  groundTruthKmh?: number;
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

export type WorkerInMessage =
  | {
      type: 'INIT';
      modelUrl: string;
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
    };

export type WorkerOutMessage =
  | { type: 'READY'; backend: string }
  | { type: 'RESULT'; detection: BallDetection | null; inferTimeMs: number }
  | { type: 'ERROR'; message: string };
