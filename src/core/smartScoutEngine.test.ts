import { describe, expect, it } from 'vitest';
import { DEFAULT_TUNNEL_CONFIGS, estimateAutomaticPitchFlight, estimatePitchFlight } from './smartScoutEngine';

const cameraAngle = 'behind_pitcher' as const;
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

describe('automatic endpoints', () => {
  const flight = detections.map((sample) => ({ ...sample, x: sample.x - 180, y: sample.y + 60 }));
  const held = { ...flight[0], timestampMs: flight[0].timestampMs - 40 };
  const last = flight[flight.length - 1];
  const stopped = [40, 80].map((dt) => ({ ...last, timestampMs: last.timestampMs + dt }));
  const auto = (samples: typeof flight) => estimateAutomaticPitchFlight({
    detections: samples, videoWidth, videoHeight, cameraAngle, distanceMeters: 18.44,
  });

  it('estimates an off-center pitch from visible movement transitions', () => {
    expect(auto([held, ...flight, ...stopped])).toMatchObject({ flightTimeMs: 400, speedKmh: 157.3 });
  });

  it('does not treat a ball disappearing or a late first detection as catch or release', () => {
    expect(auto([held, ...flight])).toBeNull();
    expect(auto([...flight, ...stopped])).toBeNull();
  });

  it('waits for finalization before using a shrinking track without a visible stop', () => {
    const shrinkingTrack = flight.map((sample, index) => ({
      ...sample, width: 11 - index * 0.65, height: 11 - index * 0.65,
    }));
    const input = { detections: shrinkingTrack, videoWidth, videoHeight, cameraAngle, distanceMeters: 18.44 };
    expect(estimateAutomaticPitchFlight({ ...input, isFinalized: false })).toBeNull();
    expect(estimateAutomaticPitchFlight({ ...input, isFinalized: true })).toMatchObject({ flightTimeMs: 400 });
  });

  it('does not bridge missing frames or accept a ball stopping at the image edge', () => {
    expect(auto([held, ...flight.filter((_, i) => i < 3 || i > 6), ...stopped])).toBeNull();
    expect(auto([held, ...flight, ...stopped].map((sample, i, all) => ({
      ...sample, x: 300 + Math.min(Math.max(i - 1, 0), all.length - 4) * 69,
    })))).toBeNull();
  });
});
