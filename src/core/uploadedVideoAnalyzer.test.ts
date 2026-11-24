import { describe, expect, it } from 'vitest';
import { pickPitchScanWindows } from './uploadedVideoAnalyzer';

describe('pickPitchScanWindows', () => {
  it('locks onto steady pitching motion burst in a 9.8s video while ignoring idle start and tripod pan end', () => {
    const samples = [];
    for (let t = 0; t <= 9.8; t += 0.1) {
      // Idle 0..4.0s, leg-lift/stride burst 4.1..5.3s (peaks at ~4.7s), ball flight 5.54..5.95s, tripod pan 8.2..9.8s (0 energy)
      const energy = t >= 4.1 && t <= 5.3 ? 0.055 : t > 5.3 && t <= 6.2 ? 0.025 : 0.0005;
      samples.push({ timeSec: Number(t.toFixed(2)), energy });
    }
    const wins = pickPitchScanWindows(9.8, samples, []);
    expect(wins.length).toBeGreaterThanOrEqual(1);
    expect(wins[0].startSec).toBeLessThanOrEqual(4.5);
    expect(wins[0].endSec).toBeGreaterThanOrEqual(6.2);
    expect(wins[0].endSec).toBeLessThan(8.2);
  });

  it('centers YOLO scan window around live 1.0x YOLO seed hit on short clip', () => {
    const wins = pickPitchScanWindows(3.2, [], [2.42]);
    expect(wins.length).toBeGreaterThanOrEqual(1);
    expect(wins[0].preferYolo).toBe(true);
    expect(wins[0].startSec).toBeLessThan(2.0);
    expect(wins[0].endSec).toBeGreaterThan(2.9);
  });
});
