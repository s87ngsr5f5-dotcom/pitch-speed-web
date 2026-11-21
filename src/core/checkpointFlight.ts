import type { BallDetection, TunnelBoxConfig } from '../types';

export function measureCheckpointFlight(
  samples: BallDetection[],
  checkpoints: TunnelBoxConfig,
  width: number,
  height: number,
  distance: number,
) {
  const release = checkpoints.releaseBox;
  const arrival = checkpoints.strikeZoneBox;
  const dx = arrival.cx - release.cx;
  const dy = arrival.cy - release.cy;
  const inBox = (sample: BallDetection, box: typeof release) =>
    Math.abs(sample.x / width - box.cx) <= box.width / 2 &&
    Math.abs(sample.y / height - box.cy) <= box.height / 2;
  const progress = (sample: BallDetection) =>
    ((sample.x / width - release.cx) * dx + (sample.y / height - release.cy) * dy) /
    (dx * dx + dy * dy);
  let start: [number, number] | undefined;
  for (let index = 1; index < samples.length; index++) {
    const before = samples[index - 1];
    const after = samples[index];
    const gap = after.timestampMs - before.timestampMs;
    if (gap <= 0 || gap > 100) continue;
    const left = progress(before);
    const right = progress(after);
    const crosses = (position: number) => left < position && right >= position;
    if (crosses(0) && [before, after].every(sample => inBox(sample, release))) {
      start = [before.timestampMs, after.timestampMs];
    }
    if (!start || !crosses(1) || ![before, after].every(sample => inBox(sample, arrival))) continue;
    const shortest = before.timestampMs - start[1];
    const longest = after.timestampMs - start[0];
    if (shortest < 120 || longest > 1600 || !Number.isFinite(distance) || distance <= 0) continue;
    return {
      averageKmh: distance * 7200 / (shortest + longest),
      minKmh: distance * 3600 / longest,
      maxKmh: distance * 3600 / shortest,
    };
  }
  return null;
}
