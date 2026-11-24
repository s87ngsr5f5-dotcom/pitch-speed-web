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

  it('tracks a dead-center optical-axis rainbow parabola with apex gap and shrinking ball', () => {
    const pw = 540;
    const ph = 960;
    const frames: MotionFrame[] = [];
    // Simulate pitch-v1/v2 style indoor throw: rises to apex (y decreases), brief 2-frame gap, then falls into catcher mitt
    const trajectory: Array<{ f: number; x: number; y: number; sz: number }> = [
      { f: 4, x: 262, y: 446, sz: 10 },
      { f: 5, x: 256, y: 441, sz: 7 },
      { f: 6, x: 254, y: 438, sz: 4 },
      { f: 7, x: 251, y: 438, sz: 2 },
      // frames 8, 9 occluded behind net / apex
      { f: 10, x: 251, y: 441, sz: 2 },
      { f: 11, x: 250, y: 443, sz: 2 },
      { f: 12, x: 251, y: 445, sz: 2 },
      { f: 13, x: 250, y: 447, sz: 2 },
      { f: 14, x: 251, y: 449, sz: 2 },
      { f: 15, x: 251, y: 451, sz: 2 },
      { f: 16, x: 252, y: 453, sz: 2 },
      { f: 17, x: 254, y: 455, sz: 2 },
      { f: 18, x: 255, y: 457, sz: 2 },
      { f: 19, x: 255, y: 459, sz: 2 },
    ];
    const byFrame = new Map(trajectory.map((pt) => [pt.f, pt]));

    for (let f = 0; f < 24; f++) {
      const pt = byFrame.get(f);
      frames.push(
        makeFrame(
          pw,
          ph,
          f * 33.37,
          45,
          pt ? (g) => drawRect(g, pw, ph, pt.x, pt.y, pt.sz, pt.sz, 235) : undefined,
          {
            // Pitcher foreground body sits right below the optical axis; head halo extends upward
            ts: f * 33.37,
            bodyBox: { x0: 120, y0: 645, x1: 315, y1: 960 },
          },
        ),
      );
    }

    const dets = extractFlyingBallTrackFromFrames(
      frames,
      pw,
      ph,
      720,
      1280,
      null,
      [{ releaseMs: 4 * 33.37, score: 24.0, throwingHand: 'R', wristX: 350, wristY: 590 }],
    );

    expect(dets.length).toBeGreaterThanOrEqual(12);
    const flightMs = dets[dets.length - 1].timestampMs - dets[0].timestampMs;
    expect(flightMs).toBeGreaterThanOrEqual(450);
    expect(flightMs).toBeLessThanOrEqual(600);
  });
});
