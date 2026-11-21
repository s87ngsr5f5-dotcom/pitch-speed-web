import { describe, expect, it } from 'vitest';
import { DEFAULT_TUNNEL_CONFIGS, estimatePitchFlight } from './smartScoutEngine';

const cameraAngle = 'behind_pitcher';
const { releaseBox, strikeZoneBox } = DEFAULT_TUNNEL_CONFIGS[cameraAngle];
const videoWidth = 1000;
const videoHeight = 600;
const checkpoints = { releaseBox, strikeZoneBox };
const detections = Array.from({ length: 11 }, (_, index) => {
  const progress = index / 10;
  return {
    x: videoWidth * (releaseBox.cx + (strikeZoneBox.cx - releaseBox.cx) * progress),
    y: videoHeight * (releaseBox.cy + (strikeZoneBox.cy - releaseBox.cy) * progress),
    width: 6,
    height: 6,
    confidence: 0.8,
    timestampMs: 3100 + index * 40,
  };
});

function estimate(samples = detections) {
  return estimatePitchFlight({
    cameraAngle,
    videoWidth,
    videoHeight,
    distanceMeters: 18.44,
    detections: samples,
    checkpoints,
  });
}

describe('video pitch speed estimate', () => {
  it('uses presentation time between both checkpoints', () => {
    expect(estimate()).toMatchObject({ flightTimeMs: 400, speedKmh: 157.3, detectionCount: 11 });
  });

  it('rejects tracks missing an endpoint and balls that do not travel down the tunnel', () => {
    expect(estimate(detections.slice(4))).toBeNull();
    expect(estimate(detections.slice(0, 7))).toBeNull();
    expect(estimate(detections.map((d) => ({ ...d, y: detections[0].y })))).toBeNull();
  });

  it('does not start the clock while the ball is held in the release box', () => {
    const heldBeforeRelease = { ...detections[0], timestampMs: 3060 };
    expect(estimate([heldBeforeRelease, ...detections])).toMatchObject({ flightTimeMs: 400 });
  });

  it('does not connect detections across a gap or cap an impossible speed', () => {
    expect(estimate(detections.filter((_, index) => index < 3 || index > 6))).toBeNull();
    expect(estimate(detections.map((d) => ({ ...d, timestampMs: d.timestampMs / 3 })))).toBeNull();
  });
});
