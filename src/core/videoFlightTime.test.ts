import { describe, expect, it } from 'vitest';
import { measureVideoFlight } from './videoFlightTime';

describe('manual video flight timing', () => {
  it('measures average speed from a known distance and elapsed source time', () => {
    expect(measureVideoFlight(2.5, 3, 15)).toEqual({ averageKmh: 108, flightSeconds: 0.5 });
  });

  it('corrects a slowed video timeline when its source multiplier is known', () => {
    expect(measureVideoFlight(3, 5, 15, 4)).toEqual({ averageKmh: 108, flightSeconds: 0.5 });
  });

  it('requires ordered, plausible timestamps and a positive distance', () => {
    expect(measureVideoFlight(3, 2, 15)).toBeNull();
    expect(measureVideoFlight(2, 2.05, 15)).toBeNull();
    expect(measureVideoFlight(2, 2.5, 0)).toBeNull();
  });
});
