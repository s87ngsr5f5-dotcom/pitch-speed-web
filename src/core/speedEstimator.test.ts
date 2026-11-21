import { describe, expect, it } from 'vitest';
import { BallDetection, CalibrationConfig, Point2D } from '../types';
import { HomographyCalculator } from './homography';
import { KalmanFilter2D } from './kalmanFilter';
import { SpeedEstimator } from './speedEstimator';

interface Vec3 {
  x: number;
  y: number;
  z: number;
}
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(v.x, v.y, v.z);
  return { x: v.x / l, y: v.y / l, z: v.z / l };
};

class VirtualObliqueCamera {
  private readonly right: Vec3;
  private readonly down: Vec3;
  private readonly forward: Vec3;

  constructor(private readonly pos: Vec3, lookAt: Vec3) {
    this.forward = norm(sub(lookAt, pos));
    this.right = norm(cross(this.forward, { x: 0, y: 1, z: 0 }));
    this.down = norm(cross(this.forward, this.right));
  }

  public project(pt: Vec3): Point2D {
    const rel = sub(pt, this.pos);
    const zc = dot(rel, this.forward);
    return {
      x: 950 * (dot(rel, this.right) / zc) + 640,
      y: 950 * (dot(rel, this.down) / zc) + 360,
    };
  }
}

function createSeededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number, sigma: number): number {
  return (
    Math.sqrt(-2 * Math.log(Math.max(1e-12, rand()))) *
    Math.cos(2 * Math.PI * rand()) *
    sigma
  );
}

const DISTANCE_M = 18.44;
const REF_HEIGHT_M = 1.5;

function createSetup() {
  const camera = new VirtualObliqueCamera(
    { x: 19.0, y: 2.5, z: 16.0 },
    { x: DISTANCE_M / 2, y: 1.0, z: 0 }
  );
  const calibration: CalibrationConfig = {
    distanceMeters: DISTANCE_M,
    referenceHeightMeters: REF_HEIGHT_M,
    imagePoints: [
      camera.project({ x: 0, y: REF_HEIGHT_M, z: 0 }),
      camera.project({ x: DISTANCE_M, y: REF_HEIGHT_M, z: 0 }),
      camera.project({ x: DISTANCE_M, y: 0, z: 0 }),
      camera.project({ x: 0, y: 0, z: 0 }),
    ],
  };
  return { camera, calibration };
}

function simulatePitch(
  camera: VirtualObliqueCamera,
  opts: {
    speedKmh: number;
    angleDeg: number;
    releaseY?: number;
    startMs?: number;
    fps?: number;
    noisePx?: number;
    jitterMs?: number;
    dropRate?: number;
    seed?: number;
  }
): BallDetection[] {
  const rand = createSeededRandom(opts.seed ?? 2026);
  const v0 = opts.speedKmh / 3.6;
  const rad = (opts.angleDeg * Math.PI) / 180;
  const vx = v0 * Math.cos(rad);
  const vy0 = v0 * Math.sin(rad);
  const dtBase = 1000 / (opts.fps ?? 60);
  const out: BallDetection[] = [];
  let elapsed = 0;

  while (elapsed < 2000) {
    const t = elapsed / 1000;
    const x = 1.2 + vx * t;
    const y = (opts.releaseY ?? 1.9) + vy0 * t - 0.5 * 9.81 * t * t;
    if (x > DISTANCE_M - 0.3 || y < 0.05) break;

    if (!(elapsed > 0 && (opts.dropRate ?? 0) > 0 && rand() < (opts.dropRate ?? 0))) {
      const px = camera.project({ x, y, z: 0 });
      out.push({
        x: px.x + (opts.noisePx ? gaussian(rand, opts.noisePx) : 0),
        y: px.y + (opts.noisePx ? gaussian(rand, opts.noisePx) : 0),
        width: 12,
        height: 12,
        confidence: 0.75 + rand() * 0.2,
        timestampMs: (opts.startMs ?? 1000) + elapsed,
      });
    }
    const jitter = opts.jitterMs ? (rand() * 2 - 1) * opts.jitterMs : 0;
    elapsed += Math.max(4, dtBase + jitter);
  }
  return out;
}

describe('PitchSpeed 18.44m 物理引擎驗證', () => {
  it('Homography 垂直切面能以次毫米精度消除斜拍高度視差', () => {
    const { camera, calibration } = createSetup();
    const h = new HomographyCalculator(calibration);
    for (const pt of [
      { x: 0, y: 0 },
      { x: 9.22, y: 0.35 },
      { x: 14.8, y: 2.8 },
    ]) {
      const px = camera.project({ x: pt.x, y: pt.y, z: 0 });
      const rec = h.pixelToWorld(px);
      expect(Math.abs(rec.x - pt.x)).toBeLessThan(1e-4);
      expect(Math.abs(rec.y - pt.y)).toBeLessThan(1e-4);
      const back = h.worldToPixel(rec);
      expect(Math.abs(back.x - px.x)).toBeLessThan(1e-3);
      expect(Math.abs(back.y - px.y)).toBeLessThan(1e-3);
    }
  });

  it('理想 140 km/h 斜拍拋物線誤差小於 0.1%', () => {
    const { camera, calibration } = createSetup();
    const est = new SpeedEstimator(calibration);
    simulatePitch(camera, { speedKmh: 140, angleDeg: -1.8 }).forEach((d) =>
      est.pushDetection(d)
    );
    const res = est.flush()!;
    expect(Math.abs(res.speedKmh - 140) / 140).toBeLessThan(0.001);
  });

  it('20 次雜訊 (σ=1.5px) + VFR 抖動 + 20% 掉幀模擬誤差皆小於 1%', () => {
    const { camera, calibration } = createSetup();
    for (let i = 1; i <= 20; i++) {
      const est = new SpeedEstimator(calibration);
      simulatePitch(camera, {
        speedKmh: 140,
        angleDeg: -1.5,
        noisePx: 1.5,
        jitterMs: 3.5,
        dropRate: 0.2,
        seed: 1000 + i * 37,
      }).forEach((d) => est.pushDetection(d));
      const res = est.flush()!;
      expect(Math.abs(res.speedKmh - 140) / 140).toBeLessThan(0.01);
    }
  });

  it('高拋物線可變高度球路 (75 km/h, +18°) 誤差小於 1%', () => {
    const { camera, calibration } = createSetup();
    const est = new SpeedEstimator(calibration);
    simulatePitch(camera, {
      speedKmh: 75,
      angleDeg: 18,
      releaseY: 1.2,
      noisePx: 1.2,
      seed: 777,
    }).forEach((d) => est.pushDetection(d));
    const res = est.flush()!;
    expect(Math.abs(res.speedKmh - 75) / 75).toBeLessThan(0.01);
  });

  it('能過濾瞬移誤判雜訊並準確切分連續兩球', () => {
    const { camera, calibration } = createSetup();
    const est = new SpeedEstimator(calibration);
    const pitch1 = simulatePitch(camera, {
      speedKmh: 140,
      angleDeg: -1.5,
      startMs: 1000,
      noisePx: 1.0,
      seed: 42,
    });
    const pitch2 = simulatePitch(camera, {
      speedKmh: 115,
      angleDeg: 2.5,
      startMs: pitch1[pitch1.length - 1].timestampMs + 500,
      noisePx: 1.0,
      seed: 99,
    });

    pitch1.forEach((d, idx) => {
      est.pushDetection(d);
      if (idx === 5) {
        est.pushDetection({
          x: 1180,
          y: 120,
          width: 15,
          height: 15,
          confidence: 0.9,
          timestampMs: d.timestampMs + 5,
        });
      }
    });

    const res1 = est.pushDetection(pitch2[0])!;
    for (let i = 1; i < pitch2.length; i++) est.pushDetection(pitch2[i]);
    const res2 = est.flush()!;

    expect(Math.abs(res1.speedKmh - 140) / 140).toBeLessThan(0.01);
    expect(Math.abs(res2.speedKmh - 115) / 115).toBeLessThan(0.01);
  });

  it('KalmanFilter2D 能穩定追蹤含重力拋物線並預測掉幀位置', () => {
    const kf = new KalmanFilter2D(9.81);
    const dt = 1 / 60;
    const vx = 38.89;
    const vy0 = 1.0;
    for (let i = 0; i < 15; i++) {
      const t = i * dt;
      const x = vx * t;
      const y = 1.8 + vy0 * t - 0.5 * 9.81 * t * t;
      if (i === 0) kf.init(x, y);
      else {
        kf.predict(dt);
        kf.update(x, y);
      }
    }
    const pred = kf.peekPredict(dt);
    const nextT = 15 * dt;
    expect(Math.abs(pred.x - vx * nextT)).toBeLessThan(0.2);
    expect(Math.abs(pred.y - (1.8 + vy0 * nextT - 0.5 * 9.81 * nextT * nextT))).toBeLessThan(0.2);
  });

  it('120fps 高幀率下遇到單幀反向框抖動不會鎖死後續軌跡', () => {
    const { camera, calibration } = createSetup();
    const est = new SpeedEstimator(calibration);
    const pitch = simulatePitch(camera, {
      speedKmh: 145,
      angleDeg: -1.5,
      fps: 120,
      noisePx: 3.5,
      seed: 20261121,
    });
    // 故意在第 8 幀注入反向抖動
    if (pitch.length > 10) {
      pitch[8] = { ...pitch[8], x: pitch[7].x - 18 };
    }
    pitch.forEach((d) => est.pushDetection(d));
    const res = est.flush()!;
    expect(res.trajectoryWorld.length).toBeGreaterThan(25);
    expect(Math.abs(res.speedKmh - 145)).toBeLessThan(1.5);
  });
});
