import { Point2D } from '../types';

export interface KalmanState {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export class KalmanFilter2D {
  private state: [number, number, number, number] = [0, 0, 0, 0];
  private P: number[][] = [];
  private initialized = false;
  private readonly g: number;
  private readonly processNoise: number;
  private readonly measNoise: number;

  constructor(gravityMps2 = 9.81, processNoise = 0.25, measNoise = 0.04) {
    this.g = gravityMps2;
    this.processNoise = processNoise;
    this.measNoise = measNoise;
    this.reset();
  }

  public reset(): void {
    this.initialized = false;
    this.state = [0, 0, 0, 0];
    this.P = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 400, 0],
      [0, 0, 0, 400],
    ];
  }

  public isInitialized(): boolean {
    return this.initialized;
  }

  public init(x: number, y: number, vx = 0, vy = 0): void {
    this.state = [x, y, vx, vy];
    this.P = [
      [0.05, 0, 0, 0],
      [0, 0.05, 0, 0],
      [0, 0, 200, 0],
      [0, 0, 0, 200],
    ];
    this.initialized = true;
  }

  public peekPredict(dtSec: number): Point2D {
    const [x, y, vx, vy] = this.state;
    return {
      x: x + vx * dtSec,
      y: y + vy * dtSec - 0.5 * this.g * dtSec * dtSec,
    };
  }

  public predict(dtSec: number): KalmanState {
    if (!this.initialized || dtSec <= 0) return this.getState();

    const [x, y, vx, vy] = this.state;
    this.state = [
      x + vx * dtSec,
      y + vy * dtSec - 0.5 * this.g * dtSec * dtSec,
      vx,
      vy - this.g * dtSec,
    ];

    const F = [
      [1, 0, dtSec, 0],
      [0, 1, 0, dtSec],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    const q = this.processNoise;
    const dt2 = dtSec * dtSec;
    const dt3 = dt2 * dtSec;
    const dt4 = dt3 * dtSec;
    const Q = [
      [(dt4 / 4) * q, 0, (dt3 / 2) * q, 0],
      [0, (dt4 / 4) * q, 0, (dt3 / 2) * q],
      [(dt3 / 2) * q, 0, dt2 * q, 0],
      [0, (dt3 / 2) * q, 0, dt2 * q],
    ];

    const FP = this.mul(F, this.P);
    const FPFt = this.mul(FP, this.transpose(F));
    this.P = FPFt.map((row, r) => row.map((v, c) => v + Q[r][c]));
    return this.getState();
  }

  public update(measX: number, measY: number): KalmanState {
    if (!this.initialized) {
      this.init(measX, measY);
      return this.getState();
    }

    const r = this.measNoise;
    const y0 = measX - this.state[0];
    const y1 = measY - this.state[1];

    const s00 = this.P[0][0] + r;
    const s01 = this.P[0][1];
    const s10 = this.P[1][0];
    const s11 = this.P[1][1] + r;
    const det = s00 * s11 - s01 * s10;
    if (Math.abs(det) < 1e-12) return this.getState();

    const invS00 = s11 / det;
    const invS01 = -s01 / det;
    const invS10 = -s10 / det;
    const invS11 = s00 / det;

    const K: [number, number][] = [0, 1, 2, 3].map((i) => [
      this.P[i][0] * invS00 + this.P[i][1] * invS10,
      this.P[i][0] * invS01 + this.P[i][1] * invS11,
    ]);

    for (let i = 0; i < 4; i++) {
      this.state[i] += K[i][0] * y0 + K[i][1] * y1;
    }

    const IKH = [0, 1, 2, 3].map((row) => [
      (row === 0 ? 1 : 0) - K[row][0],
      (row === 1 ? 1 : 0) - K[row][1],
      row === 2 ? 1 : 0,
      row === 3 ? 1 : 0,
    ]);
    this.P = this.mul(IKH, this.P);
    return this.getState();
  }

  public getState(): KalmanState {
    return {
      x: this.state[0],
      y: this.state[1],
      vx: this.state[2],
      vy: this.state[3],
    };
  }

  private mul(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const out = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let sum = 0;
        for (let k = 0; k < inner; k++) sum += A[r][k] * B[k][c];
        out[r][c] = sum;
      }
    }
    return out;
  }

  private transpose(M: number[][]): number[][] {
    return M[0].map((_, c) => M.map((row) => row[c]));
  }
}
