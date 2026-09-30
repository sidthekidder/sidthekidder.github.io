// Axis-aligned bounding box collision: pushes a point out of any
// overlapping box, then clamps it to stay within an arena boundary.
// Pure math — no THREE.js, no DOM — so it's directly unit-testable and
// reusable for both the logical leader position's navigation and every
// crowd group's per-frame building avoidance.

/**
 * Mutates `position` ({x, z}) so it no longer overlaps any box in
 * `boxes` (each {minX, maxX, minZ, maxZ}) by less than `radius`, then
 * clamps it to [-arenaHalf, arenaHalf] on both axes.
 */
export function resolveBoxCollision(position, radius, boxes, arenaHalf) {
  for (const box of boxes) {
    const closestX = Math.max(box.minX, Math.min(position.x, box.maxX));
    const closestZ = Math.max(box.minZ, Math.min(position.z, box.maxZ));
    const dx = position.x - closestX;
    const dz = position.z - closestZ;
    const distSq = dx * dx + dz * dz;
    if (distSq < radius * radius) {
      const dist = Math.sqrt(distSq) || 0.0001;
      const push = radius - dist;
      position.x += (dx / dist) * push;
      position.z += (dz / dist) * push;
    }
  }
  position.x = Math.max(-arenaHalf, Math.min(arenaHalf, position.x));
  position.z = Math.max(-arenaHalf, Math.min(arenaHalf, position.z));
}
