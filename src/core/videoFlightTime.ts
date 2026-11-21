/** Average speed between manually marked events on a video timeline. */
export function measureVideoFlight(
  releaseSeconds: number,
  arrivalSeconds: number,
  distanceMeters: number,
  timeExpansion = 1,
) {
  if (![releaseSeconds, arrivalSeconds, distanceMeters, timeExpansion].every(Number.isFinite) ||
      releaseSeconds < 0 || arrivalSeconds <= releaseSeconds ||
      distanceMeters <= 0 || timeExpansion < 1) return null;
  const flightSeconds = (arrivalSeconds - releaseSeconds) / timeExpansion;
  const averageKmh = distanceMeters / flightSeconds * 3.6;
  if (!Number.isFinite(averageKmh) || flightSeconds < 0.12 || flightSeconds > 2) return null;
  return { averageKmh, flightSeconds };
}
