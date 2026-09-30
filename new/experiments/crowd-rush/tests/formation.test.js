import { test, assertClose, assertTrue, summary } from './tiny-test.js';
import { computeFormationSlot } from '../lib/formation.js';

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

test('index 0 sits exactly at the crowd center (circle formation)', () => {
  const out = { x: 0, z: 0 };
  computeFormationSlot(0, 10, -5, 0, /* moveBlend */ 0, /* elapsedTime */ 0, out);
  assertClose(out.x, 10);
  assertClose(out.z, -5);
});

test('index 0 sits exactly at the crowd center (trail formation)', () => {
  // The leader (index 0) is ring 0, which has zero radius/depth/lateral
  // offset regardless of moveBlend or facing — it's always the crowd's
  // actual position, everyone else trails or circles around it.
  const out = { x: 0, z: 0 };
  computeFormationSlot(0, 3, 4, 1.2, /* moveBlend */ 1, /* elapsedTime */ 0, out);
  assertClose(out.x, 3);
  assertClose(out.z, 4);
});

test('later ring indices sit further from center in the circle formation', () => {
  const center = { x: 0, z: 0 };
  const near = { x: 0, z: 0 };
  const far = { x: 0, z: 0 };
  computeFormationSlot(1, 0, 0, 0, 0, 0, near); // ring 1
  computeFormationSlot(20, 0, 0, 0, 0, 0, far); // several rings out
  assertTrue(dist(far, center) > dist(near, center));
});

test('moveBlend blends continuously between circle and trail positions', () => {
  const circle = { x: 0, z: 0 };
  const trail = { x: 0, z: 0 };
  const half = { x: 0, z: 0 };
  computeFormationSlot(5, 0, 0, 0.7, 0, 0, circle);
  computeFormationSlot(5, 0, 0, 0.7, 1, 0, trail);
  computeFormationSlot(5, 0, 0, 0.7, 0.5, 0, half);
  // The halfway point should sit roughly between the two extremes, not
  // off to some unrelated position.
  const expectedX = (circle.x + trail.x) / 2;
  const expectedZ = (circle.z + trail.z) / 2;
  assertClose(half.x, expectedX, 0.1);
  assertClose(half.z, expectedZ, 0.1);
});

test('returns the same object passed as `out`', () => {
  const out = { x: 0, z: 0 };
  const returned = computeFormationSlot(2, 0, 0, 0, 0, 0, out);
  assertTrue(returned === out);
});

summary();
