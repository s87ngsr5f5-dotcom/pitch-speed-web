import { BallDetection } from '../types';
import { PitcherExclusionSnapshot, PoseReleaseEvent } from './pitcherPoseGate';

export interface MotionFrame {
  timestampMs: number;
  gray: Uint8Array;
  /** Optional pitcher torso/pants & hand snapshot in ORIGINAL video pixel coordinates */
  exclusion?: PitcherExclusionSnapshot | null;
}

interface CandidateBlob {
  x: number;
  y: number;
  w: number;
  h: number;
  area: number;
  brightness: number;
  polarity: 1 | -1;
}

interface TrackNode {
  frameIdx: number;
  timestampMs: number;
  blob: CandidateBlob;
}

/** 3x3 linear solver for quadratic least squares a*u^2 + b*u + c */
function fitQuadraticResidual(nodes: TrackNode[]): { meanErr: number; maxErr: number; ayCurv: number } {
  const n = nodes.length;
  if (n < 4) return { meanErr: 999, maxErr: 999, ayCurv: 0 };
  const t0 = nodes[0].timestampMs;
  const span = Math.max(1, nodes[n - 1].timestampMs - t0);

  // Normal equations A^T A (3x3 symmetric)
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  let bx0 = 0, bx1 = 0, bx2 = 0;
  let by0 = 0, by1 = 0, by2 = 0;

  for (let i = 0; i < n; i++) {
    const u = (nodes[i].timestampMs - t0) / span;
    const u2 = u * u;
    s0 += 1;
    s1 += u;
    s2 += u2;
    s3 += u2 * u;
    s4 += u2 * u2;
    const x = nodes[i].blob.x;
    const y = nodes[i].blob.y;
    bx0 += x; bx1 += u * x; bx2 += u2 * x;
    by0 += y; by1 += u * y; by2 += u2 * y;
  }

  const solve3 = (rhs0: number, rhs1: number, rhs2: number): [number, number, number] | null => {
    const m = [
      [s4, s3, s2, rhs2],
      [s3, s2, s1, rhs1],
      [s2, s1, s0, rhs0],
    ];
    for (let col = 0; col < 3; col++) {
      let pivot = col;
      for (let r = col + 1; r < 3; r++) {
        if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
      }
      if (Math.abs(m[pivot][col]) < 1e-10) return null;
      if (pivot !== col) {
        const tmp = m[col];
        m[col] = m[pivot];
        m[pivot] = tmp;
      }
      const div = m[col][col];
      for (let c = col; c < 4; c++) m[col][c] /= div;
      for (let r = 0; r < 3; r++) {
        if (r === col) continue;
        const f = m[r][col];
        for (let c = col; c < 4; c++) m[r][c] -= f * m[col][c];
      }
    }
    return [m[0][3], m[1][3], m[2][3]];
  };

  const cx = solve3(bx0, bx1, bx2);
  const cy = solve3(by0, by1, by2);
  if (!cx || !cy) return { meanErr: 999, maxErr: 999, ayCurv: 0 };

  let sumErr = 0;
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    const u = (nodes[i].timestampMs - t0) / span;
    const px = cx[0] * u * u + cx[1] * u + cx[2];
    const py = cy[0] * u * u + cy[1] * u + cy[2];
    const e = Math.hypot(px - nodes[i].blob.x, py - nodes[i].blob.y);
    sumErr += e;
    if (e > maxErr) maxErr = e;
  }
  return { meanErr: sumErr / n, maxErr, ayCurv: cy[0] };
}

/** Fast strided camera-pan / camera-shake rejection (identical to Desktop temporal_ball_tracker.py). */
function isCameraSteady(
  prev: Uint8Array,
  curr: Uint8Array,
  next: Uint8Array,
  procWidth: number,
  procHeight: number,
): boolean {
  const stepX = 4;
  const stepY = 4;
  let total = 0;
  let changedPrev = 0;
  let changedNext = 0;
  let highDiffCount = 0;

  for (let y = 0; y < procHeight; y += stepY) {
    const row = y * procWidth;
    for (let x = 0; x < procWidth; x += stepX) {
      const idx = row + x;
      const dp = Math.abs(curr[idx] - prev[idx]);
      const dn = Math.abs(curr[idx] - next[idx]);
      total++;
      if (dp >= 14) changedPrev++;
      if (dn >= 14) changedNext++;
      if (dp > 4) highDiffCount++;
    }
  }
  if (total === 0) return false;
  // MAD > 4 means > 50% of pixels have |diff| > 4
  if (highDiffCount / total > 0.5) return false;
  if (changedPrev / total > 0.14 || changedNext / total > 0.14) return false;
  return true;
}

/**
 * Pose-Gated, Camera-Pan-Safe 3-Frame Temporal-Difference Ballistic Ball Tracker
 * (Web PWA parity with Desktop `temporal_ball_tracker.py`).
 *
 * Recovers tiny (2-14px) flying baseballs in unseen outdoor stadiums, portrait IG
 * screen recordings (e.g. wba_shirakawa.mp4), and high-motion-blur fastballs where
 * single-frame 320px YOLO misses the ball — while rejecting camera pans, static UI icons,
 * white baseball pants, and pre-release windup arm raises.
 */
export function extractFlyingBallTrackFromFrames(
  frames: MotionFrame[],
  procWidth: number,
  procHeight: number,
  origWidth: number,
  origHeight: number,
  region?: { x: number; y: number; width: number; height: number } | null,
  releaseEvents?: PoseReleaseEvent[] | null,
): BallDetection[] {
  if (frames.length < 7 || procWidth < 32 || procHeight < 32) return [];

  // Estimate FPS & temporal stride (~33ms separation so 60/120fps balls don't self-overlap)
  const dts: number[] = [];
  for (let i = 1; i < Math.min(frames.length, 40); i++) {
    const dt = frames[i].timestampMs - frames[i - 1].timestampMs;
    if (dt > 2 && dt < 200) dts.push(dt);
  }
  dts.sort((a, b) => a - b);
  const medianDt = dts.length > 0 ? dts[Math.floor(dts.length / 2)] : 33.3;
  const estFps = Math.max(15, Math.min(240, 1000 / Math.max(4, medianDt)));
  const stride = Math.max(1, Math.round(estFps / 32));
  if (frames.length < stride * 2 + 5) return [];

  // Filter dominant release anchors (suppress minor pre-windup hand raises within 3.5s)
  const strongReleases: PoseReleaseEvent[] = (() => {
    if (!releaseEvents || releaseEvents.length === 0) return [];
    const s = releaseEvents.filter((r) => r.score >= 3.5);
    const base = s.length > 0 ? s : releaseEvents;
    if (base.length <= 1) return base;
    const kept = base.filter(
      (ev) =>
        !base.some(
          (other) =>
            other !== ev &&
            Math.abs(other.releaseMs - ev.releaseMs) <= 3500 &&
            other.score >= Math.max(10.0, ev.score * 3.2),
        ),
    );
    return kept.length > 0 ? kept : base;
  })();

  const minX = region ? Math.floor(region.x * procWidth) : Math.floor(procWidth * 0.02);
  const maxX = region ? Math.ceil((region.x + region.width) * procWidth) : Math.ceil(procWidth * 0.98);
  const minY = region ? Math.floor(region.y * procHeight) : Math.floor(procHeight * 0.06);
  const maxY = region ? Math.ceil((region.y + region.height) * procHeight) : Math.ceil(procHeight * 0.80);

  const toProcX = procWidth / Math.max(1, origWidth);
  const toProcY = procHeight / Math.max(1, origHeight);
  const dimScale = Math.min(procWidth, procHeight) / 540;
  const maxDim = Math.max(16, Math.round(34 * dimScale));
  const maxArea = Math.max(130, Math.round(420 * dimScale * dimScale));

  const totalPixels = procWidth * procHeight;
  const visited = new Uint8Array(totalPixels);
  const brightMask = new Uint8Array(totalPixels);
  const darkMask = new Uint8Array(totalPixels);
  const queue = new Int32Array(totalPixels);

  const candsByIdx: Array<{ frameIdx: number; timestampMs: number; blobs: CandidateBlob[] }> = [];

  for (let i = stride; i < frames.length - stride; i++) {
    const ts = frames[i].timestampMs;
    // If dominant release events exist, only process frames inside [release - 100ms, release + 1200ms]
    if (strongReleases.length > 0) {
      const inWin = strongReleases.some((r) => ts >= r.releaseMs - 100 && ts <= r.releaseMs + 1200);
      if (!inWin) continue;
    }

    const prev = frames[i - stride].gray;
    const curr = frames[i].gray;
    const next = frames[i + stride].gray;

    if (!isCameraSteady(prev, curr, next, procWidth, procHeight)) continue;

    brightMask.fill(0);
    darkMask.fill(0);
    let hasBright = false;
    let hasDark = false;

    for (let y = minY; y < maxY; y++) {
      const row = y * procWidth;
      for (let x = minX; x < maxX; x++) {
        const idx = row + x;
        const c = curr[idx];
        const p = prev[idx];
        const n = next[idx];
        const d1 = c - p;
        const d2 = c - n;
        if (c >= 92 && d1 >= 14 && d2 >= 14) {
          brightMask[idx] = 1;
          hasBright = true;
        } else if (p >= 170 && n >= 170 && d1 <= -18 && d2 <= -18) {
          darkMask[idx] = 1;
          hasDark = true;
        }
      }
    }

    const excl = frames[i].exclusion;
    const bodyBox = excl?.bodyBox
      ? {
          x0: excl.bodyBox.x0 * toProcX,
          y0: excl.bodyBox.y0 * toProcY,
          x1: excl.bodyBox.x1 * toProcX,
          y1: excl.bodyBox.y1 * toProcY,
        }
      : null;
    const wrist = excl?.throwingWrist
      ? {
          x: excl.throwingWrist.x * toProcX,
          y: excl.throwingWrist.y * toProcY,
          torso: excl.throwingWrist.torso * toProcY,
        }
      : null;
    const glove = excl?.gloveWrist
      ? {
          x: excl.gloveWrist.x * toProcX,
          y: excl.gloveWrist.y * toProcY,
          torso: excl.gloveWrist.torso * toProcY,
        }
      : null;

    const blobs: CandidateBlob[] = [];
    const extractFromMask = (mask: Uint8Array, polarity: 1 | -1) => {
      visited.fill(0);
      for (let y = minY; y < maxY; y++) {
        const row = y * procWidth;
        for (let x = minX; x < maxX; x++) {
          const startIdx = row + x;
          if (!mask[startIdx] || visited[startIdx]) continue;

          let head = 0;
          let tail = 0;
          queue[tail++] = startIdx;
          visited[startIdx] = 1;

          let area = 0;
          let sumX = 0;
          let sumY = 0;
          let sumB = 0;
          let bx0 = x, bx1 = x, by0 = y, by1 = y;

          while (head < tail) {
            const p = queue[head++];
            const py = Math.floor(p / procWidth);
            const px = p - py * procWidth;
            area++;
            sumX += px;
            sumY += py;
            sumB += curr[p];
            if (px < bx0) bx0 = px;
            if (px > bx1) bx1 = px;
            if (py < by0) by0 = py;
            if (py > by1) by1 = py;

            if (px > minX && mask[p - 1] && !visited[p - 1]) {
              visited[p - 1] = 1;
              queue[tail++] = p - 1;
            }
            if (px + 1 < maxX && mask[p + 1] && !visited[p + 1]) {
              visited[p + 1] = 1;
              queue[tail++] = p + 1;
            }
            if (py > minY && mask[p - procWidth] && !visited[p - procWidth]) {
              visited[p - procWidth] = 1;
              queue[tail++] = p - procWidth;
            }
            if (py + 1 < maxY && mask[p + procWidth] && !visited[p + procWidth]) {
              visited[p + procWidth] = 1;
              queue[tail++] = p + procWidth;
            }
          }

          const bw = bx1 - bx0 + 1;
          const bh = by1 - by0 + 1;
          if (area < 2 || area > maxArea || Math.max(bw, bh) > maxDim) continue;
          const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
          if (aspect > 3.6) continue;

          const cx = sumX / area;
          const cy = sumY / area;
          if (area === 2) {
            const ry = Math.min(procHeight - 1, Math.max(0, Math.round(cy)));
            const rx = Math.min(procWidth - 1, Math.max(0, Math.round(cx)));
            const cIdx = ry * procWidth + rx;
            const cd1 = curr[cIdx] - prev[cIdx];
            const cd2 = curr[cIdx] - next[cIdx];
            if (polarity === 1) {
              if (cd1 + cd2 < 34) continue;
            } else if (cd1 + cd2 > -40) {
              continue;
            }
          }

          // Exclude pitcher's torso/pants/legs unless right next to throwing wrist
          if (bodyBox && cx >= bodyBox.x0 && cx <= bodyBox.x1 && cy >= bodyBox.y0 && cy <= bodyBox.y1) {
            const nearWrist = wrist && Math.hypot(cx - wrist.x, cy - wrist.y) <= wrist.torso * 0.45;
            if (!nearWrist) continue;
          }
          // Exclude pitcher's non-throwing glove hand
          if (glove && glove.torso > 0 && Math.hypot(cx - glove.x, cy - glove.y) <= glove.torso * 0.28) {
            continue;
          }

          blobs.push({ x: cx, y: cy, w: bw, h: bh, area, brightness: sumB / area, polarity });
        }
      }
    };

    if (hasBright) extractFromMask(brightMask, 1);
    if (hasDark) extractFromMask(darkMask, -1);

    if (blobs.length > 0 && blobs.length <= 55) {
      candsByIdx.push({ frameIdx: i, timestampMs: ts, blobs });
    }
  }

  if (candsByIdx.length < 6) return [];

  // Perspective-aware ballistic linking
  const maxStepBase = Math.max(36, procWidth * 0.16);
  const minStepBase = Math.max(1.0, procWidth * 0.0020);
  const maxGapFrames = Math.max(3, Math.round(estFps * 0.105));

  let bestTrack: TrackNode[] = [];
  let bestScore = -1;

  for (let a = 0; a < candsByIdx.length - 5; a++) {
    const startEntry = candsByIdx[a];
    let matchedRel: PoseReleaseEvent | null = null;
    if (strongReleases.length > 0) {
      matchedRel =
        strongReleases.find(
          (r) => startEntry.timestampMs >= r.releaseMs - 100 && startEntry.timestampMs <= r.releaseMs + 280,
        ) ?? null;
      if (!matchedRel) continue;
    }

    for (const b0 of startEntry.blobs) {
      const seeds: Array<{ entryIdx: number; dist: number; blob: CandidateBlob }> = [];
      for (let b = a + 1; b < Math.min(candsByIdx.length, a + 4); b++) {
        const entryB = candsByIdx[b];
        const df = entryB.frameIdx - startEntry.frameIdx;
        if (df <= 0 || df > maxGapFrames) continue;
        for (const cand of entryB.blobs) {
          if (cand.polarity !== b0.polarity) continue;
          const dist = Math.hypot(cand.x - b0.x, cand.y - b0.y);
          if (dist >= minStepBase * df && dist <= maxStepBase * df) {
            seeds.push({ entryIdx: b, dist, blob: cand });
          }
        }
      }

      seeds.sort((x, y) => x.dist - y.dist);
      for (let si = 0; si < Math.min(seeds.length, 4); si++) {
        const seed = seeds[si];
        const entryB = candsByIdx[seed.entryIdx];
        const track: TrackNode[] = [
          { frameIdx: startEntry.frameIdx, timestampMs: startEntry.timestampMs, blob: b0 },
          { frameIdx: entryB.frameIdx, timestampMs: entryB.timestampMs, blob: seed.blob },
        ];
        const df0 = Math.max(1, entryB.frameIdx - startEntry.frameIdx);
        let vx = (seed.blob.x - b0.x) / df0;
        let vy = (seed.blob.y - b0.y) / df0;
        let prevSpeed = Math.hypot(vx, vy);
        let speedRatio = 0.85;

        for (let c = seed.entryIdx + 1; c < candsByIdx.length; c++) {
          const entryC = candsByIdx[c];
          const last = track[track.length - 1];
          const df = entryC.frameIdx - last.frameIdx;
          if (df > maxGapFrames) break;
          if (entryC.timestampMs - startEntry.timestampMs > 920) break;

          const predVx = vx * speedRatio;
          const predVy = vy * speedRatio;
          const predX = last.blob.x + predVx * df;
          const predY = last.blob.y + predVy * df;
          const predSpd = Math.hypot(predVx, predVy);
          const tol = Math.max(6.5, Math.min(24, predSpd * df * 0.55 + 4.5));

          let bestCand: CandidateBlob | null = null;
          let bestErr = 1e9;
          const vNorm = Math.hypot(vx, vy) + 1e-6;

          for (const cand of entryC.blobs) {
            if (cand.polarity !== b0.polarity) continue;
            const dx = cand.x - last.blob.x;
            const dy = cand.y - last.blob.y;
            const stepD = Math.hypot(dx, dy);
            if (stepD < minStepBase * 0.65 * df || stepD > maxStepBase * df) continue;
            const stepSpd = stepD / df;
            const cosSim = (dx * vx + dy * vy) / (stepD * vNorm + 1e-6);
            const isGravityApex =
              stepSpd <= 7.0 &&
              prevSpeed <= 12.0 &&
              dy >= vy * df - 2.0 &&
              dy <= vy * df + 8.5 * df &&
              Math.abs(dx - vx * df) <= Math.max(4.5, Math.abs(vx * df) * 0.9 + 3.5);
            const minSpdRatio = isGravityApex ? 0.22 : 0.38;
            if (prevSpeed > 1.8 && (stepSpd < minSpdRatio * prevSpeed || stepSpd > 1.65 * prevSpeed)) continue;
            if (cosSim < 0.68 && !isGravityApex) continue;
            const err = Math.hypot(cand.x - predX, cand.y - predY);
            const effTol = isGravityApex && df >= 2 ? Math.max(tol, 12.5) : tol;
            if (err <= effTol && err < bestErr) {
              bestErr = err;
              bestCand = cand;
            }
          }

          if (bestCand) {
            const newVx = (bestCand.x - last.blob.x) / df;
            const newVy = (bestCand.y - last.blob.y) / df;
            const newSpd = Math.hypot(newVx, newVy);
            if (prevSpeed > 1.3) {
              const instRatio = Math.max(0.55, Math.min(1.35, newSpd / prevSpeed));
              speedRatio = 0.5 * speedRatio + 0.5 * instRatio;
            }
            vx = newVx;
            vy = newVy;
            prevSpeed = newSpd;
            track.push({ frameIdx: entryC.frameIdx, timestampMs: entryC.timestampMs, blob: bestCand });
          }
        }

        if (track.length < 6) continue;

        // Trim post-catch stagnation / reversal tail
        const coreA = Math.floor(track.length / 6);
        const coreB = Math.min(track.length - 1, Math.floor((track.length * 3) / 4));
        const dirX = track[coreB].blob.x - track[coreA].blob.x;
        const dirY = track[coreB].blob.y - track[coreA].blob.y;
        const dirLen = Math.hypot(dirX, dirY);
        if (dirLen < procWidth * 0.014) continue;
        const ux = dirX / dirLen;
        const uy = dirY / dirLen;

        let endI = track.length - 1;
        while (endI >= 6) {
          const dx = track[endI].blob.x - track[endI - 1].blob.x;
          const dy = track[endI].blob.y - track[endI - 1].blob.y;
          const st = Math.hypot(dx, dy);
          const dfTail = track[endI].frameIdx - track[endI - 1].frameIdx;
          const prevSt = Math.hypot(
            track[endI - 1].blob.x - track[endI - 2].blob.x,
            track[endI - 1].blob.y - track[endI - 2].blob.y,
          );
          const isPostCatchGap = endI >= 8 && dfTail >= 2 && (st > prevSt * 1.25 || st < minStepBase * 1.4);
          const align = (dx * ux + dy * uy) / Math.max(1e-6, st);
          if (st < minStepBase * 0.75 || align < 0.10 || isPostCatchGap) endI--;
          else break;
        }
        let trimmed = track.slice(0, endI + 1);
        if (trimmed.length < 6) continue;

        // Trim pre-release backward cocking frames (preserve genuine rising-to-apex rainbow launch!)
        let startI = 0;
        while (trimmed.length - startI >= 6) {
          const dx = trimmed[startI + 1].blob.x - trimmed[startI].blob.x;
          const dy = trimmed[startI + 1].blob.y - trimmed[startI].blob.y;
          const st = Math.hypot(dx, dy);
          const align = (dx * ux + dy * uy) / Math.max(1e-6, st);
          const isRainbowLaunch = dy < 0 && Math.abs(dx) <= Math.max(6, Math.abs(dy) * 1.2);
          if (align < 0.25 && !isRainbowLaunch) startI++;
          else break;
        }
        trimmed = trimmed.slice(startI);
        if (trimmed.length < 6) continue;

        const dtMs = trimmed[trimmed.length - 1].timestampMs - trimmed[0].timestampMs;
        const span = Math.hypot(
          trimmed[trimmed.length - 1].blob.x - trimmed[0].blob.x,
          trimmed[trimmed.length - 1].blob.y - trimmed[0].blob.y,
        );
        const earlySzs = trimmed.slice(0, Math.min(3, trimmed.length)).map((n) => Math.max(n.blob.w, n.blob.h));
        const lateSzs = trimmed.slice(Math.max(0, trimmed.length - 3)).map((n) => Math.max(n.blob.w, n.blob.h));
        const startSz = Math.max(...earlySzs);
        const endSz = Math.min(...lateSzs);
        const sizeRatio = Math.max(startSz / Math.max(1, endSz), Math.max(...lateSzs) / Math.max(1, Math.min(...earlySzs)));
        const minSpan = sizeRatio >= 1.45 && trimmed.length >= 8 ? procWidth * 0.022 : procWidth * 0.055;
        if (dtMs < 300 || dtMs > 900 || span < minSpan) continue;

        // Reject tracks that end on/inside the pitcher's own body/head (e.g. post-pitch walk-off or follow-through arm),
        // while allowing distant shrunk baseballs (sizeRatio >= 1.45, endSz <= 8px) flying past above the shoulders
        const endExcl = frames[trimmed[trimmed.length - 1].frameIdx]?.exclusion;
        if (endExcl?.bodyBox) {
          const bb = endExcl.bodyBox;
          const bx0 = bb.x0 * toProcX;
          const by0 = bb.y0 * toProcY;
          const bx1 = bb.x1 * toProcX;
          const by1 = bb.y1 * toProcY;
          const bH = Math.max(20, by1 - by0);
          const endNode = trimmed[trimmed.length - 1].blob;
          const inBody = endNode.x >= bx0 && endNode.x <= bx1 && endNode.y >= by0 && endNode.y <= by1;
          const inHeadHalo =
            endNode.x >= bx0 - bH * 0.10 &&
            endNode.x <= bx1 + bH * 0.10 &&
            endNode.y >= by0 - bH * 0.26 &&
            endNode.y <= by0;
          const isDistantShrunkBall = sizeRatio >= 1.45 && endSz <= 8.0 && trimmed.length >= 8;
          if (inBody || (inHeadHalo && !isDistantShrunkBall)) {
            continue;
          }
        }

        // Reject steep upward windup hand-raise trajectories
        const upwardRise = trimmed[0].blob.y - trimmed[trimmed.length - 1].blob.y;
        const horizSpan = Math.abs(trimmed[trimmed.length - 1].blob.x - trimmed[0].blob.x);
        if (upwardRise > procHeight * 0.04 && upwardRise > horizSpan * 0.45) continue;

        // Quadratic parabola smoothness verification + gravity curvature
        const { meanErr, maxErr, ayCurv } = fitQuadraticResidual(trimmed);
        const errLimit = Math.max(3.2, procWidth * 0.009);
        if (meanErr > errLimit || maxErr > errLimit * 2.6) continue;

        // Monotonic projection along flight direction OR valid downward-curving gravity rainbow arc
        let monotonic = true;
        let prevProj = -1e9;
        for (const node of trimmed) {
          const proj = (node.blob.x - trimmed[0].blob.x) * ux + (node.blob.y - trimmed[0].blob.y) * uy;
          if (proj <= prevProj) {
            monotonic = false;
            break;
          }
          prevProj = proj;
        }
        const isGravityRainbow = ayCurv >= 4.0 && meanErr <= 2.4 && sizeRatio >= 1.4 && trimmed.length >= 8;
        if (!monotonic && !isGravityRainbow) continue;

        let score =
          trimmed.length * 22 +
          (span / procWidth) * 140 -
          meanErr * 6 +
          (sizeRatio >= 1.35 ? 18 : 0) +
          (b0.polarity === 1 ? 10 : 0);

        if (matchedRel) {
          const tDiff = Math.abs(trimmed[0].timestampMs - matchedRel.releaseMs);
          if (tDiff <= 110) score += 45 - (tDiff / 33) * 8;
          if (matchedRel.wristX !== undefined && matchedRel.wristY !== undefined) {
            const wx = matchedRel.wristX * toProcX;
            const wy = matchedRel.wristY * toProcY;
            const wDist = Math.hypot(trimmed[0].blob.x - wx, trimmed[0].blob.y - wy);
            if (wDist <= procWidth * 0.25) score += 30;
            else if (wDist > procWidth * 0.55) score -= 45;
          }
        }

        if (score > bestScore && score >= 75) {
          bestScore = score;
          bestTrack = trimmed;
        }
      }
    }
  }

  if (bestTrack.length < 6) return [];

  // Probe +1 frame after last blob for tiny catcher-mitt impact pixel
  const extTrack = [...bestTrack];
  const lastNode = extTrack[extTrack.length - 1];
  const prevNode = extTrack[extTrack.length - 2];
  const fNext = lastNode.frameIdx + 1;
  if (fNext - stride >= 0 && fNext + stride < frames.length) {
    const dfLast = Math.max(1, lastNode.frameIdx - prevNode.frameIdx);
    const predNx = lastNode.blob.x + ((lastNode.blob.x - prevNode.blob.x) / dfLast) * 0.85;
    const predNy = lastNode.blob.y + ((lastNode.blob.y - prevNode.blob.y) / dfLast) * 0.85;
    const px0 = Math.max(0, Math.round(predNx - 9));
    const px1 = Math.min(procWidth - 1, Math.round(predNx + 9));
    const py0 = Math.max(0, Math.round(predNy - 9));
    const py1 = Math.min(procHeight - 1, Math.round(predNy + 9));
    const gCurr = frames[fNext].gray;
    const gPrev = frames[fNext - stride].gray;
    const gNext = frames[fNext + stride].gray;
    let cnt = 0, sx = 0, sy = 0, bx0 = px1, bx1 = px0, by0 = py1, by1 = py0;
    for (let y = py0; y <= py1; y++) {
      const row = y * procWidth;
      for (let x = px0; x <= px1; x++) {
        const idx = row + x;
        const c = gCurr[idx], p = gPrev[idx], n = gNext[idx];
        const hit =
          lastNode.blob.polarity === 1
            ? c >= 100 && c - p >= 14 && c - n >= 14
            : p >= 165 && n >= 165 && c - p <= -14 && c - n <= -14;
        if (hit) {
          cnt++;
          sx += x;
          sy += y;
          if (x < bx0) bx0 = x;
          if (x > bx1) bx1 = x;
          if (y < by0) by0 = y;
          if (y > by1) by1 = y;
        }
      }
    }
    if (cnt >= 1 && cnt <= 36) {
      const nx = sx / cnt;
      const ny = sy / cnt;
      const nbw = bx1 - bx0 + 1;
      const nbh = by1 - by0 + 1;
      const stepX = (lastNode.blob.x - prevNode.blob.x) / dfLast;
      const stepY = (lastNode.blob.y - prevNode.blob.y) / dfLast;
      const dotFwd = (nx - lastNode.blob.x) * stepX + (ny - lastNode.blob.y) * stepY;
      const notGloveRecoil = Math.max(nbw, nbh) <= Math.max(lastNode.blob.w, lastNode.blob.h) * 1.8 + 2;
      if (dotFwd > 0 && notGloveRecoil && Math.hypot(nx - predNx, ny - predNy) <= 7.5) {
        extTrack.push({
          frameIdx: fNext,
          timestampMs: frames[fNext].timestampMs,
          blob: {
            x: nx,
            y: ny,
            w: nbw,
            h: nbh,
            area: cnt,
            brightness: 160,
            polarity: lastNode.blob.polarity,
          },
        });
      }
    }
  }

  const sx = origWidth / procWidth;
  const sy = origHeight / procHeight;
  return extTrack.map((node) => ({
    x: node.blob.x * sx,
    y: node.blob.y * sy,
    width: Math.max(4, node.blob.w * sx),
    height: Math.max(4, node.blob.h * sy),
    confidence: 0.82,
    timestampMs: node.timestampMs,
  }));
}
