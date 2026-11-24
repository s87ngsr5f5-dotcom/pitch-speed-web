import { describe, expect, it } from 'vitest';
import { extractFlyingBallTrackFromFrames, MotionFrame } from './motionBallTracker';

function makeFrame(
  w: number,
  h: number,
  tsMs: number,
  bg = 70,
  draw?: (gray: Uint8Array) => void,
  exclusion?: MotionFrame['exclusion'],
): MotionFrame {
  const gray = new Uint8Array(w * h);
  gray.fill(bg);
  if (draw) draw(gray);
  return { timestampMs: tsMs, gray, exclusion };
}

function drawRect(gray: Uint8Array, w: number, h: number, cx: number, cy: number, bw: number, bh: number, val: number) {
  const x0 = Math.max(0, Math.round(cx - bw / 2));
  const x1 = Math.min(w - 1, Math.round(cx + bw / 2));
  const y0 = Math.max(0, Math.round(cy - bh / 2));
  const y1 = Math.min(h - 1, Math.round(cy + bh / 2));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      gray[y * w + x] = val;
    }
  }
}

describe('extractFlyingBallTrackFromFrames (Hybrid Temporal+Pose Ballistic Tracker)', () => {
  it('tracks a shrinking 3D perspective fastball while ignoring static UI icons & camera pans', () => {
    const pw = 300;
    const ph = 520;
    const frames: MotionFrame[] = [];
    // Simulate wba_shirakawa-like perspective flight: 12 frames (366ms..400ms) + 6 camera-pan frames at end
    const xs = [150, 172, 189, 202, 212, 220, 227, 233, 238, 242, 246, 249];
    const ys = [190, 198, 204, 209, 213, 216, 219, 222, 224, 226, 228, 230];
    const sz = [9, 8, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3];

    for (let i = 0; i < 24; i++) {
      const ts = i * 33.33;
      const isPan = i >= 18;
      frames.push(
        makeFrame(pw, ph, ts, isPan ? 70 + (i - 17) * 28 : 70, (g) => {
          // Static bright UI icons (like IG heart/comment/red dot) on EVERY frame
          drawRect(g, pw, ph, 275, 220, 12, 12, 245);
          drawRect(g, pw, ph, 275, 290, 12, 12, 245);
          if (i >= 3 && i < 3 + xs.length) {
            const k = i - 3;
            drawRect(g, pw, ph, xs[k], ys[k], sz[k], sz[k], 235);
          }
        }),
      );
    }

    const dets = extractFlyingBallTrackFromFrames(
      frames,
      pw,
      ph,
      590,
      1280,
      null,
      [{ releaseMs: 100, throwingHand: 'R', score: 25.0, wristX: 295, wristY: 465 }],
    );

    expect(dets.length).toBeGreaterThanOrEqual(10);
    expect(dets[0].timestampMs).toBeCloseTo(100, 0);
    expect(dets[dets.length - 1].x).toBeGreaterThan(dets[0].x);
    expect(dets[dets.length - 1].timestampMs - dets[0].timestampMs).toBeGreaterThanOrEqual(300);
  });

  it('rejects steep upward windup hand-raise motions and pitcher pants blobs', () => {
    const pw = 300;
    const ph = 520;
    const frames: MotionFrame[] = [];
    for (let i = 0; i < 18; i++) {
      const ts = i * 33.33;
      frames.push(
        makeFrame(
          pw,
          ph,
          ts,
          65,
          (g) => {
            // Upward hand raise during windup (y decreases steeply)
            if (i >= 2 && i <= 14) {
              const k = i - 2;
              drawRect(g, pw, ph, 130 - k * 4, 240 - k * 10, 6, 6, 230);
            }
            // Moving white pants leg inside pitcher bodyBox
            if (i >= 2 && i <= 14) {
              const k = i - 2;
              drawRect(g, pw, ph, 90 + k * 5, 340, 7, 7, 235);
            }
          },
          {
            ts,
            bodyBox: { x0: 60 * 2, y0: 280 * 2, x1: 180 * 2, y1: 420 * 2 },
          },
        ),
      );
    }

    const dets = extractFlyingBallTrackFromFrames(frames, pw, ph, 600, 1040, null, null);
    expect(dets).toHaveLength(0);
  });
});
