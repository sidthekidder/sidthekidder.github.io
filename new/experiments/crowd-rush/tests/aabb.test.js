import { test, assertClose, assertTrue, summary } from './tiny-test.js';
import { resolveBoxCollision } from '../lib/aabb.js';

const ARENA_HALF = 30;

test('a position outside every box and inside the arena is unchanged', () => {
  const boxes = [{ minX: -4, maxX: 4, minZ: -4, maxZ: 4 }];
  const position = { x: 10, z: 10 };
  resolveBoxCollision(position, 0.3, boxes, ARENA_HALF);
  assertClose(position.x, 10);
  assertClose(position.z, 10);
});

test('a position just outside a box, within radius of its edge, gets pushed clear', () => {
  const boxes = [{ minX: -4, maxX: 4, minZ: -4, maxZ: 4 }];
  const radius = 0.5;
  // 0.2 outside the box's right edge — within radius, so it should push.
  const position = { x: 4.2, z: 0 };
  resolveBoxCollision(position, radius, boxes, ARENA_HALF);
  // Pushed further out along +x to maxX + radius.
  assertClose(position.x, 4 + radius, 0.01);
  assertClose(position.z, 0, 0.01);
});

test('a position already fully inside a box is left alone (known limitation)', () => {
  // The closest-point formula clamps position into the box's range on
  // each axis, so for any position already inside, dx and dz are both
  // exactly 0 — there's no direction to push in, not just at the box's
  // center. This is a pre-existing characteristic of the algorithm (it's
  // designed to catch something approaching from outside, within radius
  // of the surface, which is the only way this game ever calls it) not
  // something this test suite is asserting as correct in general.
  const boxes = [{ minX: -4, maxX: 4, minZ: -4, maxZ: 4 }];
  const position = { x: 1, z: 0 }; // inside the box, off-center
  resolveBoxCollision(position, 0.5, boxes, ARENA_HALF);
  assertClose(position.x, 1);
  assertClose(position.z, 0);
});

test('a position is clamped to the arena boundary', () => {
  const position = { x: 1000, z: -1000 };
  resolveBoxCollision(position, 0.3, [], ARENA_HALF);
  assertClose(position.x, ARENA_HALF);
  assertClose(position.z, -ARENA_HALF);
});

test('only overlapping boxes push the position; others are ignored', () => {
  const boxes = [
    { minX: -4, maxX: 4, minZ: -4, maxZ: 4 }, // edge is within radius
    { minX: 20, maxX: 24, minZ: 20, maxZ: 24 }, // far away, irrelevant
  ];
  const position = { x: 4.2, z: 0 }; // just outside the first box
  resolveBoxCollision(position, 0.5, boxes, ARENA_HALF);
  assertTrue(
    position.x > 4.2,
    'should have been pushed further from the overlapping box, not toward the distant one'
  );
});

summary();
