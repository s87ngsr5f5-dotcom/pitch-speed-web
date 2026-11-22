import { BallDetection } from '../types';

export interface MotionFrame {
  timestampMs: number;
  gray: Uint8Array;
}

interface CandidateBlob {
  x: number;
  y: number;
  w: number;
  h: number;
  brightness: number;
}

/**
 * 3-Frame Temporal Difference Flying-Ball Tracker (SmartScout / TrackNet-style fallback).
 * Isolates fast-moving small bright objects (flying baseballs, even with 147 km/h motion blur)
 * while ignoring static backgrounds, UI icons, and slow windup preparation motions.
 */
export function extractFlyingBallTrackFromFrames(
  frames: MotionFrame[],
  procWidth: number,
  procHeight: number,
  origWidth: number,
  origHeight: number,
  region?: { x: number; y: number; width: number; height: number } | null
): BallDetection[] {
  if (frames.length < 5 || procWidth < 32 || procHeight < 32) return [];

  const minX = region ? Math.floor(region.x * procWidth) : Math.floor(procWidth * 0.05);
  const maxX = region ? Math.ceil((region.x + region.width) * procWidth) : Math.ceil(procWidth * 0.86);
  const minY = region ? Math.floor(region.y * procHeight) : Math.floor(procHeight * 0.10);
  const maxY = region ? Math.ceil((region.y + region.height) * procHeight) : Math.ceil(procHeight * 0.84);

  const visited = new Uint8Array(procWidth * procHeight);
  const mask = new Uint8Array(procWidth * procHeight);
  const queue = new Int32Array(procWidth * procHeight);

  const candsByIdx: Array<{ frameIdx: number; timestampMs: number; blobs: CandidateBlob[] }> = [];

  for (let i = 1; i < frames.length - 1; i++) {
    const prev = frames[i - 1].gray;
    const curr = frames[i].gray;
    const next = frames[i + 1].gray;

    mask.fill(0);
    visited.fill(0);

    for (let y = minY; y < maxY; y++) {
      const row = y * procWidth;
      for (let x = minX; x < maxX; x++) {
        const idx = row + x;
        const c = curr[idx];
        if (c < 130) continue;
        const d1 = c - prev[idx];
        const d2 = c - next[idx];
        // Ball must be noticeably brighter than BOTH previous and next frames (flying white ball!)
        if (d1 >= 22 && d2 >= 22) {
          mask[idx] = 1;
        }
      }
    }

    const blobs: CandidateBlob[] = [];
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
        let bx0 = x;
        let bx1 = x;
        let by0 = y;
        let by1 = y;

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
        const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
        // Small isolated bright moving object (rejects large pitcher body/limbs & 1px sensor noise)
        if (area >= 4 && area <= 220 && Math.max(bw, bh) <= 22 && aspect <= 3.2) {
          blobs.push({
            x: sumX / area,
            y: sumY / area,
            w: bw,
            h: bh,
            brightness: sumB / area,
          });
        }
      }
    }

    if (blobs.length > 0 && blobs.length <= 25) {
      candsByIdx.push({ frameIdx: i, timestampMs: frames[i].timestampMs, blobs });
    }
  }

  // Link the longest smooth fast-moving linear/parabolic arc (true pitch flight, 320ms..860ms)
  type Node = { frameIdx: number; timestampMs: number; blob: CandidateBlob };
  let bestTrack: Node[] = [];
  let bestScore = -1;

  for (let a = 0; a < candsByIdx.length; a++) {
    const startEntry = candsByIdx[a];
    for (const startBlob of startEntry.blobs) {
      const track: Node[] = [{ frameIdx: startEntry.frameIdx, timestampMs: startEntry.timestampMs, blob: startBlob }];
      let lastF = startEntry.frameIdx;
      let lx = startBlob.x;
      let ly = startBlob.y;
      let vx: number | null = null;
      let vy: number | null = null;

      for (let b = a + 1; b < candsByIdx.length; b++) {
        const entry = candsByIdx[b];
        const df = entry.frameIdx - lastF;
        if (df > 2) break;
        const elapsedMs = entry.timestampMs - startEntry.timestampMs;
        if (elapsedMs > 860) break;

        let chosen: CandidateBlob | null = null;
        let chosenErr = 1e9;
        for (const cand of entry.blobs) {
          const stepDist = Math.hypot(cand.x - lx, cand.y - ly);
          if (vx === null || vy === null) {
            if (stepDist >= 2.0 && stepDist <= 55 * df && stepDist < chosenErr) {
              chosenErr = stepDist;
              chosen = cand;
            }
          } else {
            const predX = lx + vx * df;
            const predY = ly + vy * df;
            const err = Math.hypot(cand.x - predX, cand.y - predY);
            if (err <= 14 * df && stepDist >= 1.2 && stepDist <= 55 * df && err < chosenErr) {
              chosenErr = err;
              chosen = cand;
            }
          }
        }

        if (chosen) {
          vx = (chosen.x - lx) / df;
          vy = (chosen.y - ly) / df;
          lastF = entry.frameIdx;
          lx = chosen.x;
          ly = chosen.y;
          track.push({ frameIdx: entry.frameIdx, timestampMs: entry.timestampMs, blob: chosen });
        }
      }

      if (track.length >= 6) {
        // 裁切首尾慢速段（濾除出手前擺臂與進手套後停滯／反彈），只保留真正高速飛行核心段
        const stepSpeeds: number[] = [];
        for (let k = 0; k < track.length - 1; k++) {
          const df = Math.max(1, track[k + 1].frameIdx - track[k].frameIdx);
          stepSpeeds.push(
            Math.hypot(
              track[k + 1].blob.x - track[k].blob.x,
              track[k + 1].blob.y - track[k].blob.y
            ) / df
          );
        }
        const sortedSpd = [...stepSpeeds].sort((u, v) => u - v);
        const vRef = sortedSpd[Math.floor(sortedSpd.length * 0.75)] ?? 0;
        const speedFloor = Math.max(1.8, vRef * 0.42);

        // 計算主飛行方向向量（取中段高速區）
        const coreA = Math.floor(track.length * 0.25);
        const coreB = Math.max(coreA + 1, Math.floor(track.length * 0.75));
        const dirDx = track[coreB].blob.x - track[coreA].blob.x;
        const dirDy = track[coreB].blob.y - track[coreA].blob.y;
        const dirLen = Math.hypot(dirDx, dirDy) || 1;
        const ux = dirDx / dirLen;
        const uy = dirDy / dirLen;

        const isFastForward = (k: number) => {
          const dx = track[k + 1].blob.x - track[k].blob.x;
          const dy = track[k + 1].blob.y - track[k].blob.y;
          const len = Math.hypot(dx, dy) || 1;
          const align = (dx * ux + dy * uy) / len;
          return stepSpeeds[k] >= speedFloor && align >= 0.35;
        };

        let s = 0;
        let e = track.length - 1;
        while (s + 5 <= e && !isFastForward(s)) s++;
        while (e - 5 >= s && !isFastForward(e - 1)) e--;

        const trimmed = track.slice(s, e + 1);
        if (trimmed.length >= 6) {
          const dtMs = trimmed[trimmed.length - 1].timestampMs - trimmed[0].timestampMs;
          const span = Math.hypot(
            trimmed[trimmed.length - 1].blob.x - trimmed[0].blob.x,
            trimmed[trimmed.length - 1].blob.y - trimmed[0].blob.y
          );
          // 真正高速飛行時間（360ms..740ms 對應約 85–158 km/h）且具明顯位移
          if (dtMs >= 360 && dtMs <= 740 && span >= procWidth * 0.04) {
            const avgVel = span / Math.max(1, dtMs);
            const score = trimmed.length * 8 + avgVel * 120 + span * 0.25;
            if (score > bestScore) {
              bestScore = score;
              bestTrack = trimmed;
            }
          }
        }
      }
    }
  }

  if (bestTrack.length < 6) return [];

  const sx = origWidth / procWidth;
  const sy = origHeight / procHeight;
  return bestTrack.map((node) => ({
    x: node.blob.x * sx,
    y: node.blob.y * sy,
    width: Math.max(4, node.blob.w * sx),
    height: Math.max(4, node.blob.h * sy),
    confidence: 0.78,
    timestampMs: node.timestampMs,
  }));
}
