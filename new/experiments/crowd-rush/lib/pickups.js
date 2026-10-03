// Pure pickup-effect math and group-count mutation. No THREE.js, no DOM —
// safe to import from both the browser game and plain Node test files.

// A flat delta scales by count bucket rather than a fixed amount, so it
// stays a meaningful swing at any crowd size instead of becoming noise on
// a large one.
export function scaledPickupDelta(count) {
  if (count < 30) return 10;
  if (count < 80) return 20;
  if (count < 200) return 40;
  return 80;
}

export function applyAdd(count) {
  return count + scaledPickupDelta(count);
}

export function applySub(count) {
  return count - scaledPickupDelta(count);
}

export function applyMult(count) {
  return Math.round(count * 1.5);
}

export function applyDiv(count) {
  return Math.round(count / 1.5);
}

/**
 * Percentage `applyFn`'s effect represents relative to `referenceCount` —
 * for a pickup's label, so it reads the same way regardless of a flat
 * type's current bucket (e.g. "+33%" instead of a raw "+10").
 */
export function pickupPercentText(applyFn, referenceCount) {
  const after = Math.max(1, applyFn(referenceCount));
  const pct = Math.round(((after - referenceCount) / referenceCount) * 100);
  return `${pct >= 0 ? '+' : ''}${pct}%`;
}

/**
 * Appends `amount` worth of new individuals to `groups` (as groupSize-sized
 * chunks, same shape as initial crowd spawning), scattered with a small
 * random jitter around (x, z), so a crowd that grows from a pickup gains
 * that many actual rendered members, not just a bigger count.
 */
export function addGroupsAtPosition(groups, amount, x, z, groupSize) {
  let placed = 0;
  while (placed < amount) {
    const size = Math.min(groupSize, amount - placed);
    groups.push({
      position: { x: x + (Math.random() - 0.5) * 0.6, z: z + (Math.random() - 0.5) * 0.6 },
      velocity: { x: 0, z: 0 },
      size,
    });
    placed += size;
  }
}

/**
 * Removes `amount` worth of individuals from the end of `groups`,
 * shrinking or popping groups as needed, so a shrinking crowd loses that
 * many actual rendered members. Mutates `groups` in place.
 */
export function removeCountFromGroups(groups, amount) {
  let remaining = amount;
  while (remaining > 0 && groups.length > 0) {
    const last = groups[groups.length - 1];
    if (last.size <= remaining) {
      remaining -= last.size;
      groups.pop();
    } else {
      last.size -= remaining;
      remaining = 0;
    }
  }
}
