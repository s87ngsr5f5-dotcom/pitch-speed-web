import { BallDetection, CalibrationConfig, Point2D, SpeedResult } from '../types';
import { HomographyCalculator } from './homography';
import { KalmanFilter2D } from './kalmanFilter';

interface WorldSample {
  x: number;
  y: number;
  timestampMs: number;
  confidence: number;
}

export interface EstimatorOptions {
  minConfidence?: number;
  minSamples?: number;
  maxGapMs?: number;
  maxPlausibleKmh?: number;
  minPlausibleKmh?: number;
  gravityMps2?: number;
  marginMeters?: number;
  airDragCompensation?: number;
}

const DEFAULT_OPTIONS: Required<EstimatorOptions> = {
  minConfidence: 0.35,
  minSamples: 4,
  maxGapMs: 180,
  maxPlausibleKmh: 200,
  minPlausibleKmh: 25,
  gravityMps2: 9.81,
  marginMeters: 1.5,
  airDragCompensation: 1.0,
};

export class SpeedEstimator {
  private homography: HomographyCalculator;
  private kalman: KalmanFilter2D;
  private config: CalibrationConfig;
  private options: Required<EstimatorOptions>;
  private activeTrack: WorldSample[] = [];
  private lastResult: SpeedResult | null = null;

  constructor(config: CalibrationConfig, options: EstimatorOptions = {}) {
    this.config = config;
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.homography = new HomographyCalculator(config);
    this.kalman = new KalmanFilter2D(this.options.gravityMps2);
  }

  public updateCalibration(config: CalibrationConfig): void {
    this.config = config;
    this.homography = new HomographyCalculator(config);
    this.reset();
  }

  public reset(): void {
    this.activeTrack = [];
    this.lastResult = null;
    this.kalman.reset();
  }

  public getHomography(): HomographyCalculator {
    return this.homography;
  }

  public predictNextWorldPoint(dtMs: number): Point2D | null {
    if (!this.kalman.isInitialized() || dtMs <= 0) return null;
    return this.kalman.peekPredict(dtMs / 1000);
  }

  public pushDetection(detection: BallDetection): SpeedResult | null {
    if (detection.confidence < this.options.minConfidence) return null;

    const worldPt = this.homography.pixelToWorld({ x: detection.x, y: detection.y });
    if (!Number.isFinite(worldPt.x) || !Number.isFinite(worldPt.y)) return null;

    const { distanceMeters } = this.config;
    const { marginMeters } = this.options;
    if (
      worldPt.x < -marginMeters ||
      worldPt.x > distanceMeters + marginMeters ||
      worldPt.y < -0.5 ||
      worldPt.y > 6.0
    ) {
      return null;
    }

    const sample: WorldSample = {
      x: worldPt.x,
      y: worldPt.y,
      timestampMs: detection.timestampMs,
      confidence: detection.confidence,
    };

    if (this.activeTrack.length > 0) {
      const prev = this.activeTrack[this.activeTrack.length - 1];
      const dtMs = sample.timestampMs - prev.timestampMs;
      if (dtMs <= 0) return null;

      if (dtMs > this.options.maxGapMs) {
        const finalized = this.computeCurrentSpeed();
        this.activeTrack = [sample];
        this.kalman.reset();
        this.kalman.init(sample.x, sample.y);
        return finalized;
      }

      const distM = Math.hypot(sample.x - prev.x, sample.y - prev.y);
      const instantKmh = (distM / (dtMs / 1000)) * 3.6;
      if (instantKmh > this.options.maxPlausibleKmh * 1.35) return null;

      if (this.activeTrack.length >= 2) {
        const refDx = prev.x - this.activeTrack[0].x;
        const currDx = sample.x - prev.x;
        if (Math.abs(refDx) > 0.25 && Math.sign(refDx) * currDx < -0.12) return null;
      }

      this.kalman.predict(dtMs / 1000);
      this.kalman.update(sample.x, sample.y);
    } else {
      this.kalman.init(sample.x, sample.y);
    }

    this.activeTrack.push(sample);

    if (this.activeTrack.length >= this.options.minSamples) {
      const result = this.computeCurrentSpeed();
      if (result) this.lastResult = result;
      return result;
    }
    return null;
  }

  public flush(): SpeedResult | null {
    const result = this.computeCurrentSpeed();
    this.activeTrack = [];
    this.kalman.reset();
    if (result) this.lastResult = result;
    return result;
  }

  public getLastResult(): SpeedResult | null {
    return this.lastResult;
  }

  private computeCurrentSpeed(): SpeedResult | null {
    const n = this.activeTrack.length;
    if (n < this.options.minSamples) return null;

    const t0 = this.activeTrack[0].timestampMs;
    const flightTimeMs = this.activeTrack[n - 1].timestampMs - t0;
    if (flightTimeMs < 20) return null;

    const timesSec = this.activeTrack.map((s) => (s.timestampMs - t0) / 1000);
    const xs = this.activeTrack.map((s) => s.x);
    const g = this.options.gravityMps2;
    const ysCompensated = this.activeTrack.map(
      (s, i) => s.y + 0.5 * g * timesSec[i] * timesSec[i]
    );

    const regX = this.linearRegression(timesSec, xs);
    const regY = this.linearRegression(timesSec, ysCompensated);

    const rawSpeedMps = Math.hypot(regX.slope, regY.slope);
    const speedKmh = rawSpeedMps * this.options.airDragCompensation * 3.6;

    if (speedKmh < this.options.minPlausibleKmh || speedKmh > this.options.maxPlausibleKmh) {
      return null;
    }

    const avgDetConf = this.activeTrack.reduce((acc, s) => acc + s.confidence, 0) / n;
    const sampleScore = Math.min(1, n / 10);
    const r2Clamped = Math.max(0, Math.min(1, regX.rSquared));
    const confidenceScore = Number(
      (r2Clamped * 0.65 + sampleScore * 0.2 + avgDetConf * 0.15).toFixed(3)
    );

    return {
      speedKmh: Number(speedKmh.toFixed(1)),
      speedMph: Number((speedKmh * 0.621371).toFixed(1)),
      flightTimeMs: Math.round(flightTimeMs),
      confidenceScore,
      trajectoryWorld: this.activeTrack.map((s) => ({
        x: Number(s.x.toFixed(3)),
        y: Number(s.y.toFixed(3)),
      })),
    };
  }

  private linearRegression(
    t: number[],
    y: number[]
  ): { slope: number; intercept: number; rSquared: number } {
    const n = t.length;
    let sumT = 0;
    let sumY = 0;
    for (let i = 0; i < n; i++) {
      sumT += t[i];
      sumY += y[i];
    }
    const meanT = sumT / n;
    const meanY = sumY / n;

    let ssTT = 0;
    let ssTY = 0;
    let ssYY = 0;
    for (let i = 0; i < n; i++) {
      const dt = t[i] - meanT;
      const dy = y[i] - meanY;
      ssTT += dt * dt;
      ssTY += dt * dy;
      ssYY += dy * dy;
    }
    if (ssTT < 1e-12) return { slope: 0, intercept: meanY, rSquared: 0 };

    const slope = ssTY / ssTT;
    const intercept = meanY - slope * meanT;
    let ssRes = 0;
    for (let i = 0; i < n; i++) {
      const err = y[i] - (intercept + slope * t[i]);
      ssRes += err * err;
    }
    const rSquared = ssYY < 1e-12 ? 1 : 1 - ssRes / ssYY;
    return { slope, intercept, rSquared };
  }
}
