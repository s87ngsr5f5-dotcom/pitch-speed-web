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
  area: number;
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

  // 預設鎖定在「投球飛行空中走廊」(上方 14% ~ 54%)，徹底排除畫面下半部的投手白球鞋、腿部與字幕干擾
  const minX = region ? Math.floor(region.x * procWidth) : Math.floor(procWidth * 0.08);
  const maxX = region ? Math.ceil((region.x + region.width) * procWidth) : Math.ceil(procWidth * 0.84);
  const minY = region ? Math.floor(region.y * procHeight) : Math.floor(procHeight * 0.14);
  const maxY = region ? Math.ceil((region.y + region.height) * procHeight) : Math.ceil(procHeight * 0.54);

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
        // 嚴格限定飛行棒球尺寸（4~110px、邊長 <= 15px），排除大型白球鞋／手套／手臂
        if (area >= 4 && area <= 110 && Math.max(bw, bh) <= 15 && aspect <= 3.0) {
          blobs.push({
            x: sumX / area,
            y: sumY / area,
            w: bw,
            h: bh,
            area,
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
        // 1) 峰值面積出手裁切（Peak-Area Release）：
        //    投手後方視角在真正出手瞬間球與手最靠近鏡頭（面積達峰值），出手後持續遠離縮小；
        //    若前 45% 出現明顯面積成長峰值，裁掉先前尚未出手的揮臂起手影格
        let trimmed = track;
        const firstHalfLen = Math.max(3, Math.floor(trimmed.length * 0.45));
        let peakIdx = 0;
        for (let k = 1; k < firstHalfLen; k++) {
          if (trimmed[k].blob.area > trimmed[peakIdx].blob.area) peakIdx = k;
        }
        if (
          peakIdx > 0 &&
          trimmed[peakIdx].blob.area >= trimmed[0].blob.area * 1.8 &&
          trimmed.length - peakIdx >= 6
        ) {
          trimmed = trimmed.slice(Math.max(0, peakIdx - 1));
        }

        // 2) 裁切進手套後的反彈／停滯尾段（僅移除方向反轉或近乎靜止點）
        if (trimmed.length >= 6) {
          const coreA = Math.max(0, Math.floor(trimmed.length * 0.15));
          const coreB = Math.min(trimmed.length - 1, Math.floor(trimmed.length * 0.75));
          const dirDx = trimmed[coreB].blob.x - trimmed[coreA].blob.x;
          const dirDy = trimmed[coreB].blob.y - trimmed[coreA].blob.y;
          const dirLen = Math.hypot(dirDx, dirDy) || 1;
          const ux = dirDx / dirLen;
          const uy = dirDy / dirLen;

          let e = trimmed.length - 1;
          while (e >= 6) {
            const dx = trimmed[e].blob.x - trimmed[e - 1].blob.x;
            const dy = trimmed[e].blob.y - trimmed[e - 1].blob.y;
            const st = Math.hypot(dx, dy);
            const align = st > 1e-6 ? (dx * ux + dy * uy) / st : 0;
            if (st < 1.6 || align < -0.1) {
              e--;
            } else {
              break;
            }
          }
          trimmed = trimmed.slice(0, e + 1);
        }

        if (trimmed.length >= 6) {
          const dtMs = trimmed[trimmed.length - 1].timestampMs - trimmed[0].timestampMs;
          const span = Math.hypot(
            trimmed[trimmed.length - 1].blob.x - trimmed[0].blob.x,
            trimmed[trimmed.length - 1].blob.y - trimmed[0].blob.y
          );
          // 真正高速飛行時間（410ms..680ms 對應約 92–153 km/h）且具明顯位移
          if (dtMs >= 410 && dtMs <= 680 && span >= procWidth * 0.04) {
            const avgVel = span / Math.max(1, dtMs);
            const score = trimmed.length * 10 + avgVel * 120;
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
