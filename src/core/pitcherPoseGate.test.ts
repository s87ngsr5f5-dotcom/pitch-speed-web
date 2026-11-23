import { describe, expect, it } from 'vitest';
import { PitcherPoseGate, PoseKeypoints } from './pitcherPoseGate';

/** 產生一個 RHP 骨架：肩 y=400、髖 y=600（軀幹 200px），可調整右手腕位置與抬腿 */
function pose(rWrist: [number, number], kneeY = 800): PoseKeypoints {
  const kp: [number, number, number][] = Array.from({ length: 17 }, () => [0, 0, 0]);
  kp[5] = [460, 400, 0.9]; // L shoulder
  kp[6] = [540, 400, 0.9]; // R shoulder
  kp[9] = [450, 560, 0.9]; // L wrist（手套手，不動）
  kp[10] = [rWrist[0], rWrist[1], 0.9];
  kp[11] = [470, 600, 0.9];
  kp[12] = [530, 600, 0.9];
  kp[13] = [470, kneeY, 0.9];
  kp[14] = [530, 800, 0.9];
  return kp;
}

const DT = 33;

function feedIdle(gate: PitcherPoseGate, startMs: number, frames: number) {
  let t = startMs;
  for (let i = 0; i < frames; i++, t += DT) gate.ingest(pose([540, 560]), t);
  return t;
}

/** 完整投球：抬腿 + 拉臂（~0.8s）→ 高位快速甩臂 → 隨球 */
function feedPitch(gate: PitcherPoseGate, startMs: number) {
  let t = startMs;
  // 抬腿期：膝蓋抬到髖附近，手臂慢慢往後上方拉
  for (let i = 0; i < 24; i++, t += DT) {
    gate.ingest(pose([540 + i * 4, 560 - i * 8], 620), t);
  }
  // 出手：手腕在肩膀上方快速往前甩
  const whip: [number, number][] = [[600, 330], [520, 300], [400, 320], [300, 420]];
  let releaseApprox = t;
  for (const p of whip) {
    gate.ingest(pose(p), t);
    releaseApprox = t;
    t += DT;
  }
  // 隨球
  for (let i = 0; i < 20; i++, t += DT) gate.ingest(pose([300, 560]), t);
  return { t, releaseApprox };
}

describe('PitcherPoseGate', () => {
  it('blocks measurement during idle / windup and opens only after release', () => {
    const gate = new PitcherPoseGate();
    const t0 = feedIdle(gate, 0, 30);
    expect(gate.releases).toHaveLength(0);
    expect(gate.isMeasurementAllowed(t0 - DT)).toBe(false);

    const { releaseApprox } = feedPitch(gate, t0);
    expect(gate.releases).toHaveLength(1);
    const rel = gate.releases[0];
    expect(rel.throwingHand).toBe('R');
    expect(Math.abs(rel.releaseMs - releaseApprox)).toBeLessThanOrEqual(4 * DT);
    // 出手前（抬腿期）不測速，出手後窗口內測速
    expect(gate.isMeasurementAllowed(t0 + 5 * DT)).toBe(false);
    expect(gate.isMeasurementAllowed(rel.releaseMs + 300)).toBe(true);
    expect(gate.isMeasurementAllowed(rel.releaseMs + 2000)).toBe(false);
  });

  it('rejects a whip-like spike with no preceding windup (false release)', () => {
    const gate = new PitcherPoseGate();
    // 一開始就快速揮手（例如畫面一開始的雜訊），前面沒有任何準備動作
    const spike: [number, number][] = [[540, 560], [600, 330], [420, 300], [300, 420], [300, 560]];
    let t = 0;
    for (const p of spike) {
      gate.ingest(pose(p), t);
      t += DT;
    }
    feedIdle(gate, t, 10);
    expect(gate.releases).toHaveLength(0);
  });

  it('filters ball samples to the release window and passes through when pitcher is not visible', () => {
    const gate = new PitcherPoseGate();
    const t0 = feedIdle(gate, 0, 30);
    feedPitch(gate, t0);
    const rel = gate.releases[0].releaseMs;
    const balls = [rel - 500, rel - 200, rel, rel + 100, rel + 400, rel + 700].map((timestampMs) => ({ timestampMs }));
    const kept = gate.filter(balls).map((b) => b.timestampMs);
    expect(kept).toEqual([rel, rel + 100, rel + 400, rel + 700]);

    const blind = new PitcherPoseGate();
    for (let t = 0; t < 1000; t += DT) blind.ingest(null, t);
    expect(blind.filter(balls)).toHaveLength(balls.length);
  });
});
