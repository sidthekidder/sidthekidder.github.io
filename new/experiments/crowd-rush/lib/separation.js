import { forEachNearby } from './spatialHash.js';

// Boids-style separation: push an entity away from every nearby neighbor
// within `radius`, harder the closer it is. This is what gives
// jostle/knockback in Crowd Rush — groups (regardless of which crowd owns
// them) push apart from close neighbors, with no rigid-body solver
// involved. See https://en.wikipedia.org/wiki/Boids — this is the
// "separation" rule; the game doesn't need alignment/cohesion since
// computeFormationSlot already provides an explicit destination.

/**
 * Writes the combined separation force on `entity` (from every neighbor
 * within `radius`, found via the spatial hash) into `out` ({x, z}) and
 * returns it.
 */
export function computeSeparation(entity, hash, cellSize, radius, strength, out) {
  out.x = 0;
  out.z = 0;
  forEachNearby(hash, entity.position.x, entity.position.z, cellSize, (other) => {
    if (other === entity) return;
    const dx = entity.position.x - other.position.x;
    const dz = entity.position.z - other.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz) || 0.001;
    if (dist < radius) {
      const push = (radius - dist) / radius;
      out.x += (dx / dist) * push;
      out.z += (dz / dist) * push;
    }
  });
  out.x *= strength;
  out.z *= strength;
  return out;
}
