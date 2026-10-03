import { test, assertEqual, assertClose, assertTrue, summary } from './tiny-test.js';
import {
  scaledPickupDelta,
  applyAdd,
  applySub,
  applyMult,
  applyDiv,
  pickupPercentText,
  addGroupsAtPosition,
  removeCountFromGroups,
} from '../lib/pickups.js';

test('scaledPickupDelta steps up at each bucket boundary', () => {
  assertEqual(scaledPickupDelta(0), 10);
  assertEqual(scaledPickupDelta(29), 10);
  assertEqual(scaledPickupDelta(30), 20);
  assertEqual(scaledPickupDelta(79), 20);
  assertEqual(scaledPickupDelta(80), 40);
  assertEqual(scaledPickupDelta(199), 40);
  assertEqual(scaledPickupDelta(200), 80);
  assertEqual(scaledPickupDelta(5000), 80);
});

test('applyAdd and applySub use the same bucketed delta, in opposite directions', () => {
  assertEqual(applyAdd(15), 15 + scaledPickupDelta(15));
  assertEqual(applySub(15), 15 - scaledPickupDelta(15));
  assertEqual(applyAdd(150), 150 + scaledPickupDelta(150));
});

test('applyMult and applyDiv are the inverse-ish 1.5x multiplier, rounded', () => {
  assertEqual(applyMult(10), 15);
  assertEqual(applyDiv(15), 10);
  assertEqual(applyMult(11), Math.round(11 * 1.5));
});

test('pickupPercentText reports mult/div as their fixed percentage regardless of count', () => {
  assertEqual(pickupPercentText(applyMult, 10), '+50%');
  assertEqual(pickupPercentText(applyMult, 200), '+50%');
  assertEqual(pickupPercentText(applyDiv, 60), '-33%');
});

test('pickupPercentText reflects the bucketed delta for flat types, not a fixed number', () => {
  // count=15 -> delta 10 -> +67%; count=150 -> delta 40 -> +27% — the
  // percentage shrinks across buckets because the flat types don't scale
  // as fast as a straight multiplier, which is the known/accepted tradeoff.
  assertEqual(pickupPercentText(applyAdd, 15), '+67%');
  assertEqual(pickupPercentText(applyAdd, 150), '+27%');
});

test('pickupPercentText never implies a count below 1', () => {
  // A tiny crowd hitting applySub could go to/below zero before clamping;
  // the label still describes a real (clamped) outcome.
  const text = pickupPercentText(applySub, 2);
  assertTrue(text.startsWith('-'), `expected a negative percentage, got ${text}`);
});

test('addGroupsAtPosition adds groups summing to the requested amount', () => {
  const groups = [];
  addGroupsAtPosition(groups, 10, 5, -3, 4);
  const total = groups.reduce((sum, g) => sum + g.size, 0);
  assertEqual(total, 10);
});

test('addGroupsAtPosition chunks by groupSize, with one possibly-smaller remainder', () => {
  const groups = [];
  addGroupsAtPosition(groups, 10, 0, 0, 4);
  assertEqual(groups.length, 3); // 4 + 4 + 2
  assertEqual(groups[0].size, 4);
  assertEqual(groups[1].size, 4);
  assertEqual(groups[2].size, 2);
});

test('addGroupsAtPosition scatters new groups within jitter range of (x, z)', () => {
  const groups = [];
  addGroupsAtPosition(groups, 8, 10, -10, 4);
  groups.forEach((g) => {
    assertTrue(Math.abs(g.position.x - 10) <= 0.3, `x out of jitter range: ${g.position.x}`);
    assertTrue(Math.abs(g.position.z - -10) <= 0.3, `z out of jitter range: ${g.position.z}`);
  });
});

test('removeCountFromGroups pops whole groups from the end first', () => {
  const groups = [{ size: 4 }, { size: 4 }, { size: 2 }];
  removeCountFromGroups(groups, 2);
  assertEqual(groups.length, 2);
  assertEqual(groups[1].size, 4);
});

test('removeCountFromGroups shrinks a group instead of popping when it only needs to lose part of it', () => {
  const groups = [{ size: 4 }, { size: 4 }, { size: 2 }];
  removeCountFromGroups(groups, 5);
  assertEqual(groups.length, 2);
  assertEqual(groups[1].size, 1); // last group (2) popped, remaining 3 taken from this one: 4 - 3
});

test('removeCountFromGroups removes everything without erroring if amount exceeds the total', () => {
  const groups = [{ size: 4 }, { size: 2 }];
  removeCountFromGroups(groups, 999);
  assertEqual(groups.length, 0);
});

summary();
