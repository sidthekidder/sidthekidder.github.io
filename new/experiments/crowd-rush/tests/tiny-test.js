// Minimal dependency-free test harness — no framework install needed,
// consistent with the rest of this site's no-build-step approach. Run a
// test file directly with `node tests/<name>.test.js`; call summary() at
// the end so CI/a shell script gets a real exit code.

let passed = 0;
let failed = 0;

export function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok  ${name}`);
  } catch (err) {
    failed++;
    console.error(`FAIL  ${name}`);
    console.error(`      ${err.message}`);
  }
}

export function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `expected ${expected}, got ${actual}`);
  }
}

export function assertClose(actual, expected, tolerance, message) {
  const tol = tolerance === undefined ? 1e-6 : tolerance;
  if (Math.abs(actual - expected) > tol) {
    throw new Error(
      message || `expected ~${expected} (±${tol}), got ${actual}`
    );
  }
}

export function assertTrue(value, message) {
  if (!value) {
    throw new Error(message || `expected a truthy value, got ${value}`);
  }
}

export function summary() {
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exitCode = 1;
  }
}
