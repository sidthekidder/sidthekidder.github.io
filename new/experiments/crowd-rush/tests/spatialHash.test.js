import { test, assertEqual, assertTrue, summary } from './tiny-test.js';
import { cellKey, buildSpatialHash, forEachNearby } from '../lib/spatialHash.js';

test('cellKey produces a distinct string per cell coordinate', () => {
  assertEqual(cellKey(0, 0), '0:0');
  assertEqual(cellKey(-1, 2), '-1:2');
  assertTrue(cellKey(1, 2) !== cellKey(2, 1), 'cellKey should not be symmetric');
});

test('buildSpatialHash buckets entities by their cell', () => {
  const cellSize = 1;
  const a = { position: { x: 0.2, z: 0.2 } }; // cell (0,0)
  const b = { position: { x: 0.8, z: 0.1 } }; // cell (0,0)
  const c = { position: { x: 5.5, z: 5.5 } }; // cell (5,5)
  const hash = buildSpatialHash([a, b, c], cellSize);

  const nearOrigin = hash.get(cellKey(0, 0));
  assertTrue(nearOrigin.includes(a) && nearOrigin.includes(b));
  assertTrue(!nearOrigin.includes(c));

  const farCell = hash.get(cellKey(5, 5));
  assertTrue(farCell.includes(c));
});

test('forEachNearby finds entities within the 3x3 cell neighborhood', () => {
  const cellSize = 1;
  const center = { position: { x: 0.5, z: 0.5 } }; // cell (0,0)
  const adjacent = { position: { x: 1.5, z: 0.5 } }; // cell (1,0) - neighbor
  const far = { position: { x: 10.5, z: 10.5 } }; // cell (10,10) - not a neighbor
  const hash = buildSpatialHash([center, adjacent, far], cellSize);

  const found = [];
  forEachNearby(hash, 0.5, 0.5, cellSize, (entity) => found.push(entity));

  assertTrue(found.includes(center));
  assertTrue(found.includes(adjacent));
  assertTrue(!found.includes(far));
});

test('forEachNearby on an empty region calls the callback zero times', () => {
  const hash = buildSpatialHash([], 1);
  let calls = 0;
  forEachNearby(hash, 100, 100, 1, () => {
    calls++;
  });
  assertEqual(calls, 0);
});

summary();
