// Pure size/speed formulas shared across Crowd Rush's crowd simulation.
// No THREE.js, no DOM — safe to import from both the browser game and
// plain Node test files.

/** How "big" a crowd of `count` soldiers reads for capture range. */
export function crowdRadius(count) {
  return 0.6 + Math.sqrt(count) * 0.35;
}

// Road gaps between buildings are 4 units wide. Capture range should keep
// growing with crowd size, but the radius used to push the *logical*
// leader position off buildings must stay under half that gap, or a big
// crowd gets wedged between two buildings at once.
export function buildingCollisionRadius(count) {
  return Math.min(crowdRadius(count), 1.3);
}

// Base steering speed once within CATCHUP_START of the target slot.
// Beyond that, speed ramps up with distance (capped at CATCHUP_MAX_SPEED)
// so a group that's fallen behind sprints back rather than crawling at
// the same speed as everyone else already in formation. There is
// deliberately no teleport/snap fallback anywhere in this game — however
// far behind a group falls, steerSpeedForDistance's ramp is the only
// recovery mechanism, so it always moves there continuously.
export const BASE_STEER_SPEED = 6;
export const CATCHUP_START = 3;
export const CATCHUP_RATE = 2.5;
export const CATCHUP_MAX_SPEED = 14;

export function steerSpeedForDistance(dist) {
  if (dist <= CATCHUP_START) return BASE_STEER_SPEED;
  return Math.min(
    CATCHUP_MAX_SPEED,
    BASE_STEER_SPEED + (dist - CATCHUP_START) * CATCHUP_RATE
  );
}

/** Camera shake strength for absorbing a crowd of `absorbedCount`, capped so huge absorptions don't over-shake. */
export function shakeMagnitudeForAbsorb(absorbedCount) {
  return Math.min(0.9, absorbedCount * 0.03);
}
