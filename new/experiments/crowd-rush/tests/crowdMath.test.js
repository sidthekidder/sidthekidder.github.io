import { test, assertEqual, assertClose, assertTrue, summary } from './tiny-test.js';
import {
  crowdRadius,
  buildingCollisionRadius,
  steerSpeedForDistance,
  BASE_STEER_SPEED,
  CATCHUP_START,
  CATCHUP_MAX_SPEED,
} from '../lib/crowdMath.js';

test('crowdRadius(0) is the base radius with no size contribution', () => {
  assertClose(crowdRadius(0), 0.6);
});

test('crowdRadius grows monotonically with count', () => {
  assertTrue(crowdRadius(50) > crowdRadius(10));
  assertTrue(crowdRadius(10) > crowdRadius(1));
});

test('buildingCollisionRadius matches crowdRadius below the cap', () => {
  // crowdRadius(4) = 0.6 + sqrt(4)*0.35 = 1.3 exactly, the cap boundary.
  assertClose(buildingCollisionRadius(0), crowdRadius(0));
  assertClose(buildingCollisionRadius(4), 1.3);
});

test('buildingCollisionRadius caps at 1.3 for large crowds', () => {
  assertTrue(crowdRadius(100) > 1.3, 'test assumption: crowdRadius(100) exceeds the cap');
  assertClose(buildingCollisionRadius(100), 1.3);
});

test('steerSpeedForDistance stays at base speed within CATCHUP_START', () => {
  assertEqual(steerSpeedForDistance(0), BASE_STEER_SPEED);
  assertEqual(steerSpeedForDistance(CATCHUP_START), BASE_STEER_SPEED);
});

test('steerSpeedForDistance ramps up beyond CATCHUP_START', () => {
  const speed = steerSpeedForDistance(CATCHUP_START + 1);
  assertTrue(speed > BASE_STEER_SPEED);
});

test('steerSpeedForDistance never exceeds CATCHUP_MAX_SPEED', () => {
  assertEqual(steerSpeedForDistance(1000), CATCHUP_MAX_SPEED);
});

test('steerSpeedForDistance is monotonically non-decreasing', () => {
  let previous = steerSpeedForDistance(0);
  for (let dist = 0.5; dist <= 20; dist += 0.5) {
    const speed = steerSpeedForDistance(dist);
    assertTrue(speed >= previous, `speed dropped at dist=${dist}`);
    previous = speed;
  }
});

summary();
