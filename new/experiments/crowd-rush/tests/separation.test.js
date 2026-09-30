import { test, assertEqual, assertTrue, summary } from './tiny-test.js';
import { buildSpatialHash } from '../lib/spatialHash.js';
import { computeSeparation } from '../lib/separation.js';

const CELL_SIZE = 1.2;
const RADIUS = 0.9;
const STRENGTH = 5;

test('two entities within radius push apart from each other', () => {
  const a = { position: { x: 0, z: 0 } };
  const b = { position: { x: 0.3, z: 0 } }; // well within RADIUS
  const hash = buildSpatialHash([a, b], CELL_SIZE);
  const out = { x: 0, z: 0 };

  computeSeparation(a, hash, CELL_SIZE, RADIUS, STRENGTH, out);

  // b is to a's +x side, so a should be pushed in -x.
  assertTrue(out.x < 0, `expected a pushed toward -x, got ${out.x}`);
  assertEqual(out.z, 0);
});

test('entities further apart than radius produce no separation force', () => {
  const a = { position: { x: 0, z: 0 } };
  const b = { position: { x: 5, z: 5 } }; // far beyond RADIUS
  const hash = buildSpatialHash([a, b], CELL_SIZE);
  const out = { x: 1, z: 1 }; // pre-filled to confirm computeSeparation resets it

  computeSeparation(a, hash, CELL_SIZE, RADIUS, STRENGTH, out);

  assertEqual(out.x, 0);
  assertEqual(out.z, 0);
});

test('an entity never separates from itself', () => {
  const a = { position: { x: 0, z: 0 } };
  const hash = buildSpatialHash([a], CELL_SIZE);
  const out = { x: 0, z: 0 };

  computeSeparation(a, hash, CELL_SIZE, RADIUS, STRENGTH, out);

  assertEqual(out.x, 0);
  assertEqual(out.z, 0);
});

test('closer neighbors push harder than more distant ones', () => {
  const a = { position: { x: 0, z: 0 } };
  const close = { position: { x: 0.2, z: 0 } };
  const near = { position: { x: 0.7, z: 0 } };

  const hashClose = buildSpatialHash([a, close], CELL_SIZE);
  const hashNear = buildSpatialHash([a, near], CELL_SIZE);
  const outClose = { x: 0, z: 0 };
  const outNear = { x: 0, z: 0 };

  computeSeparation(a, hashClose, CELL_SIZE, RADIUS, STRENGTH, outClose);
  computeSeparation(a, hashNear, CELL_SIZE, RADIUS, STRENGTH, outNear);

  assertTrue(
    Math.abs(outClose.x) > Math.abs(outNear.x),
    'a closer neighbor should push harder'
  );
});

summary();
