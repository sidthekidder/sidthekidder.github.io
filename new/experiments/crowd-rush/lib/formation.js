// Formation layout math: where a given individual index within a crowd
// belongs, blending between a gathered circle and a wedge trailing
// behind a facing direction. Pure function of its inputs (including
// elapsedTime, passed explicitly rather than read from a module-level
// clock) so it's directly unit-testable.

/**
 * Where soldier `index` (within a crowd of any size) belongs, blending
 * between a gathered circle (moveBlend 0) and a wedge trailing behind the
 * facing direction (moveBlend 1). This is a target for steering, not a
 * direct position write — separation from neighbors still gets the final
 * say each frame in the caller.
 *
 * Writes into `out` ({x, z}) instead of returning a fresh object and
 * returns it too, so callers can either use the return value or ignore it
 * and read `out` — this avoids a fresh allocation on every call, which
 * matters since this runs once per group every single frame in the game's
 * hot path.
 */
export function computeFormationSlot(
  index,
  centerX,
  centerZ,
  facingAngle,
  moveBlend,
  elapsedTime,
  out
) {
  let ring = 0;
  let ringStart = 0;
  let ringCapacity = 1;
  while (index >= ringStart + ringCapacity) {
    ringStart += ringCapacity;
    ring++;
    ringCapacity = ring * 6;
  }
  const posInRing = index - ringStart;
  const baseAngle = (posInRing / ringCapacity) * Math.PI * 2;
  const seed = index * 12.9898;
  const wobble = Math.sin(elapsedTime * 2.4 + seed) * 0.07;

  const ringRadius = ring * 0.55;
  const circleX = centerX + Math.cos(baseAngle) * (ringRadius + wobble);
  const circleZ = centerZ + Math.sin(baseAngle) * (ringRadius + wobble);

  const forwardX = Math.sin(facingAngle);
  const forwardZ = Math.cos(facingAngle);
  const rightX = Math.cos(facingAngle);
  const rightZ = -Math.sin(facingAngle);
  const depthOffset = ring * 0.42 + Math.cos(baseAngle) * (ring * 0.12) + wobble;
  const lateralOffset = Math.sin(baseAngle) * (ring * 0.5) + wobble;
  const trailX = centerX - forwardX * depthOffset + rightX * lateralOffset;
  const trailZ = centerZ - forwardZ * depthOffset + rightZ * lateralOffset;

  out.x = circleX + (trailX - circleX) * moveBlend;
  out.z = circleZ + (trailZ - circleZ) * moveBlend;
  return out;
}
