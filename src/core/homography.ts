import { CalibrationConfig, Point2D } from '../types';

type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number
];

export class HomographyCalculator {
  private H: Mat3;
  private HInv: Mat3;

  constructor(config: CalibrationConfig) {
    const worldPoints = HomographyCalculator.buildVerticalPlaneWorldPoints(
      config.distanceMeters,
      config.referenceHeightMeters
    );
    this.H = HomographyCalculator.computeHomography(config.imagePoints, worldPoints);
    this.HInv = HomographyCalculator.invert3x3(this.H);
  }

  public static buildVerticalPlaneWorldPoints(
    distanceMeters: number,
    referenceHeightMeters: number
  ): [Point2D, Point2D, Point2D, Point2D] {
    if (distanceMeters <= 0 || referenceHeightMeters <= 0) {
      throw new Error('校正距離與參考高度必須大於 0 公尺');
    }
    return [
      { x: 0, y: referenceHeightMeters },
      { x: distanceMeters, y: referenceHeightMeters },
      { x: distanceMeters, y: 0 },
      { x: 0, y: 0 },
    ];
  }

  public pixelToWorld(pt: Point2D): Point2D {
    return HomographyCalculator.applyMatrix(this.H, pt);
  }

  public worldToPixel(pt: Point2D): Point2D {
    return HomographyCalculator.applyMatrix(this.HInv, pt);
  }

  private static applyMatrix(M: Mat3, pt: Point2D): Point2D {
    const w = M[6] * pt.x + M[7] * pt.y + M[8];
    if (Math.abs(w) < 1e-12) return { x: NaN, y: NaN };
    return {
      x: (M[0] * pt.x + M[1] * pt.y + M[2]) / w,
      y: (M[3] * pt.x + M[4] * pt.y + M[5]) / w,
    };
  }

  private static computeHomography(
    src: readonly Point2D[],
    dst: readonly Point2D[]
  ): Mat3 {
    const { normalized: normSrc, T: Tsrc } = this.normalizePoints(src);
    const { normalized: normDst, TInv: TdstInv } = this.normalizePoints(dst);

    const A: number[][] = [];
    const b: number[] = [];
    for (let i = 0; i < 4; i++) {
      const { x: u, y: v } = normSrc[i];
      const { x: X, y: Y } = normDst[i];
      A.push([u, v, 1, 0, 0, 0, -u * X, -v * X]);
      b.push(X);
      A.push([0, 0, 0, u, v, 1, -u * Y, -v * Y]);
      b.push(Y);
    }

    const h = this.solveLinearSystem(A, b);
    const HNorm: Mat3 = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
    const H = this.multiply3x3(this.multiply3x3(TdstInv, HNorm), Tsrc);

    if (Math.abs(H[8]) > 1e-12) {
      const scale = 1 / H[8];
      return H.map((val) => val * scale) as Mat3;
    }
    return H;
  }

  private static normalizePoints(points: readonly Point2D[]): {
    normalized: Point2D[];
    T: Mat3;
    TInv: Mat3;
  } {
    let cx = 0;
    let cy = 0;
    for (const p of points) {
      cx += p.x;
      cy += p.y;
    }
    cx /= points.length;
    cy /= points.length;

    let meanDist = 0;
    for (const p of points) {
      meanDist += Math.hypot(p.x - cx, p.y - cy);
    }
    meanDist /= points.length;
    if (meanDist < 1e-9) throw new Error('校正點彼此過於接近或重疊');

    const s = Math.SQRT2 / meanDist;
    const normalized = points.map((p) => ({
      x: (p.x - cx) * s,
      y: (p.y - cy) * s,
    }));
    const T: Mat3 = [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
    const TInv: Mat3 = [1 / s, 0, cx, 0, 1 / s, cy, 0, 0, 1];
    return { normalized, T, TInv };
  }

  private static solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = b.length;
    const aug = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      let maxVal = Math.abs(aug[col][col]);
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(aug[row][col]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxVal < 1e-12) throw new Error('四個標定點共線或構成退化四邊形');
      if (maxRow !== col) {
        const temp = aug[col];
        aug[col] = aug[maxRow];
        aug[maxRow] = temp;
      }
      const pivot = aug[col][col];
      for (let row = col + 1; row < n; row++) {
        const factor = aug[row][col] / pivot;
        for (let k = col; k <= n; k++) {
          aug[row][k] -= factor * aug[col][k];
        }
      }
    }

    const x = new Array<number>(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = aug[i][n];
      for (let j = i + 1; j < n; j++) sum -= aug[i][j] * x[j];
      x[i] = sum / aug[i][i];
    }
    return x;
  }

  private static multiply3x3(A: Mat3, B: Mat3): Mat3 {
    const out = new Array<number>(9).fill(0);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        out[r * 3 + c] =
          A[r * 3 + 0] * B[0 * 3 + c] +
          A[r * 3 + 1] * B[1 * 3 + c] +
          A[r * 3 + 2] * B[2 * 3 + c];
      }
    }
    return out as Mat3;
  }

  private static invert3x3(m: Mat3): Mat3 {
    const [a, b, c, d, e, f, g, h, i] = m;
    const c00 = e * i - f * h;
    const c01 = -(d * i - f * g);
    const c02 = d * h - e * g;
    const c10 = -(b * i - c * h);
    const c11 = a * i - c * g;
    const c12 = -(a * h - b * g);
    const c20 = b * f - c * e;
    const c21 = -(a * f - c * d);
    const c22 = a * e - b * d;
    const det = a * c00 + b * c01 + c * c02;
    if (Math.abs(det) < 1e-12) throw new Error('單應性矩陣不可逆');
    const inv = 1 / det;
    return [
      c00 * inv, c10 * inv, c20 * inv,
      c01 * inv, c11 * inv, c21 * inv,
      c02 * inv, c12 * inv, c22 * inv,
    ];
  }
}
