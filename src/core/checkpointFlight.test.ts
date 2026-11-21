import { expect, it } from 'vitest';
import { measureCheckpointFlight } from './checkpointFlight';

const checkpoints = {
  releaseBox: { cx: 0.2, cy: 0.5, width: 0.1, height: 0.2 },
  strikeZoneBox: { cx: 0.8, cy: 0.5, width: 0.1, height: 0.2 },
};
const sample = (x: number, timestampMs: number) => ({
  x, y: 50, timestampMs, width: 1, height: 1, confidence: 0.9,
});
it('times both crossings across a gap in ball detections', () => {
  expect(measureCheckpointFlight(
    [sample(19, 100), sample(21, 133), sample(79, 600), sample(81, 633)],
    checkpoints, 100, 100, 15,
  )).toMatchObject({ averageKmh: 108 });
});
it('requires nearby detections on both sides of the arrival checkpoint', () => {
  expect(measureCheckpointFlight(
    [sample(19, 100), sample(21, 133), sample(81, 633)],
    checkpoints, 100, 100, 15,
  )).toBeNull();
});
