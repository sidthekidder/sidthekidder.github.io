# Experimental Showcase Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hidden, no-build static page at `/new` on the existing Jekyll site with a scroll-animated landing hero + tile grid, linking to two standalone experiments: a WebGL2 matrix-rain shader and a Three.js audio-reactive visualizer.

**Architecture:** Plain HTML/CSS/JS (ES modules), Three.js and GSAP loaded from a CDN (esm.sh) — no npm/build tooling. Everything lives under `/new/` at the repo root, which Jekyll passes through untouched. Each experiment is a fully standalone page under `/new/experiments/<name>/`.

**Tech Stack:** HTML5, CSS3, vanilla JS ES modules, GSAP 3 + ScrollTrigger (esm.sh CDN), Three.js r165 (esm.sh CDN), WebGL2 (raw GLSL for the matrix demo), Web Audio API.

**Spec:** `docs/superpowers/specs/2026-09-28-experimental-showcase-design.md`

## Global Constraints

- No build step: files are edited directly and served as-is; Three.js/GSAP are imported via CDN ES module URLs (esm.sh), not installed via npm.
- Everything lives under `/new/` at the repo root; no changes to the existing Jekyll site, its nav, `_config.yml`, posts, or other pages.
- No backend, no persistence — all state (audio buffers, animation state, shader uniforms) is client-side and ephemeral for the page's lifetime.
- No automated test suite — verification is manual, in a real browser, for every task.
- Each experiment independently checks its own requirements (e.g. WebGL2 availability) and shows a fallback message instead of crashing; the landing page must not assume any experiment "just works."
- The page must not be linked from existing site navigation (hidden by omission, not by access control).
- Local verification before every commit: run `python3 -m http.server 8000` from the repo root, then visit `http://localhost:8000/new/...` in a browser.

## Review Focus

- WebGL2 unavailable in the visitor's browser → matrix demo must show a readable fallback message, not a blank canvas or console error.
- Microphone permission denied on the audio visualizer → must fall back to the file-upload path with a clear status message, not a stuck "Requesting microphone…" state.
- No microphone hardware present at all → `getUserMedia` rejects (e.g. `NotFoundError`); must land on the same fallback path as a denial, not an uncaught rejection.
- Uploaded file that isn't valid decodable audio → `decodeAudioData` rejection must surface a user-facing error message, not silent failure or a frozen "Decoding audio…" state.
- Narrow/mobile viewport → hero, tile grid, and both experiment canvases must resize/reflow without horizontal overflow or unusably small tap targets.

---

### Task 1: Shared visual foundation + landing page shell

**Files:**
- Create: `new/style.css`
- Create: `new/index.html`

**Interfaces:**
- Produces: CSS custom properties `--bg`, `--fg`, `--accent`, `--accent-2`, `--muted` (used by every later page); class names `.hero`, `.hero-title`, `.hero-subtitle`, `.scroll-cue`, `.tile-grid`, `.tile`, `.tile.is-visible`, `.tile-title`, `.tile-desc`, `.fallback-message`, `.back-link` (consumed by Tasks 2-5).

- [ ] **Step 1: Write `new/style.css`**

```css
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --bg: #0a0a0f;
  --fg: #f4f4f8;
  --accent: #7cfcae;
  --accent-2: #ff4fd8;
  --muted: #8a8a99;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--fg);
  font-family: 'Space Grotesk', system-ui, sans-serif;
  overflow-x: hidden;
}

.hero {
  position: relative;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2rem;
}

.hero-title {
  font-size: clamp(3rem, 12vw, 8rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.hero-subtitle {
  margin-top: 1rem;
  font-size: clamp(1rem, 2vw, 1.5rem);
  color: var(--muted);
}

.scroll-cue {
  position: absolute;
  bottom: 2rem;
  color: var(--muted);
  font-size: 0.9rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  animation: bob 2s ease-in-out infinite;
}

@keyframes bob {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(8px); }
}

.tile-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
  padding: 4rem 2rem 8rem;
  max-width: 1100px;
  margin: 0 auto;
}

.tile {
  display: block;
  text-decoration: none;
  color: var(--fg);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2rem;
  background: rgba(255, 255, 255, 0.03);
  opacity: 0;
  transform: translateY(30px);
}

.tile.is-visible {
  opacity: 1;
  transform: translateY(0);
  transition: opacity 0.6s ease, transform 0.6s ease;
}

.tile:hover {
  border-color: var(--accent);
  background: rgba(255, 255, 255, 0.06);
  transform: translateY(-4px);
  transition: transform 0.3s ease, border-color 0.3s ease, background 0.3s ease;
}

.tile-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.tile-desc {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.4;
}

.fallback-message {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  text-align: center;
  padding: 2rem;
  color: var(--muted);
  font-size: 1.1rem;
}

.back-link {
  position: fixed;
  top: 1.5rem;
  left: 1.5rem;
  color: var(--fg);
  text-decoration: none;
  font-size: 0.9rem;
  z-index: 10;
  opacity: 0.7;
}

.back-link:hover {
  opacity: 1;
}
```

- [ ] **Step 2: Write `new/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex, nofollow">
<title>Experiments — Siddhartha Sahai</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
  <section class="hero">
    <h1 class="hero-title">EXPERIMENTS</h1>
    <p class="hero-subtitle">A playground of small, weird, technical things.</p>
    <div class="scroll-cue">scroll ↓</div>
  </section>

  <section class="tile-grid" id="tile-grid">
    <a class="tile" href="experiments/matrix/index.html" data-tile>
      <h2 class="tile-title">Matrix Rain</h2>
      <p class="tile-desc">A generative WebGL2 shader, reactive to your cursor.</p>
    </a>
    <a class="tile" href="experiments/audio-visualizer/index.html" data-tile>
      <h2 class="tile-title">Audio Visualizer</h2>
      <p class="tile-desc">Speak, sing, or upload a track — watch it move in 3D.</p>
    </a>
  </section>
</body>
</html>
```

- [ ] **Step 3: Verify locally**

Run: `python3 -m http.server 8000` from the repo root.
Visit: `http://localhost:8000/new/`
Expected: A full-screen dark hero with the gradient "EXPERIMENTS" title, subtitle, and a bobbing "scroll ↓" cue; scrolling down reveals two bordered tiles in a grid (they'll render, but their links 404 until Tasks 2-3 exist — that's expected for now).

- [ ] **Step 4: Commit**

```bash
git add new/style.css new/index.html
git commit -m "feat: scaffold hidden experiments landing page shell"
```

---

### Task 2: Matrix rain WebGL2 shader experiment

**Files:**
- Create: `new/experiments/matrix/index.html`
- Create: `new/experiments/matrix/matrix.js`

**Interfaces:**
- Consumes: `new/style.css` classes `.fallback-message`, `.back-link` from Task 1.

- [ ] **Step 1: Write `new/experiments/matrix/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex, nofollow">
<title>Matrix Rain — Experiments</title>
<link rel="stylesheet" href="../../style.css">
<style>
  body { overflow: hidden; }
  canvas { display: block; width: 100vw; height: 100vh; }
</style>
</head>
<body>
  <a class="back-link" href="../../index.html">← back</a>
  <canvas id="gl-canvas"></canvas>
  <script type="module" src="matrix.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `new/experiments/matrix/matrix.js`**

```javascript
const canvas = document.getElementById('gl-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  document.body.innerHTML =
    '<div class="fallback-message">This experiment needs WebGL2, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
} else {
  const vertexSource = `#version 300 es
  layout(location = 0) in vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }`;

  const fragmentSource = `#version 300 es
  precision highp float;
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform vec2 u_mouse;
  out vec4 outColor;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float cols = 60.0;
    float col = floor(uv.x * cols);
    float speed = 0.3 + hash(vec2(col, 0.0)) * 0.7;
    float offset = hash(vec2(col, 1.0)) * 10.0;
    float y = fract(uv.y + u_time * speed + offset);

    float dist = length(uv - u_mouse);
    float mouseGlow = smoothstep(0.25, 0.0, dist);

    float charNoise = step(0.5, hash(vec2(col, floor((uv.y + u_time * speed) * 40.0))));
    float brightness = pow(1.0 - y, 4.0) * charNoise;
    brightness += mouseGlow * 0.3;

    vec3 green = vec3(0.1, 1.0, 0.4) * brightness;
    outColor = vec4(green, 1.0);
  }`;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  const positions = new Float32Array([-1, -1, 3, -1, -1, 3]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
  const timeLoc = gl.getUniformLocation(program, 'u_time');
  const mouseLoc = gl.getUniformLocation(program, 'u_mouse');

  let mouseX = 0.5;
  let mouseY = 0.5;

  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX / window.innerWidth;
    mouseY = 1.0 - e.clientY / window.innerHeight;
  });

  function resize() {
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener('resize', resize);
  resize();

  function render(timeMs) {
    const time = timeMs * 0.001;
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniform2f(resolutionLoc, canvas.width, canvas.height);
    gl.uniform1f(timeLoc, time);
    gl.uniform2f(mouseLoc, mouseX, mouseY);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}
```

- [ ] **Step 3: Verify locally**

Run: `python3 -m http.server 8000` from the repo root (if not already running).
Visit: `http://localhost:8000/new/experiments/matrix/index.html`
Expected: A fullscreen green falling-code pattern that brightens near the mouse cursor; resizing the window keeps it fullscreen without distortion; the "← back" link returns to the landing page.

- [ ] **Step 4: Verify the WebGL2 fallback path**

Temporarily change `canvas.getContext('webgl2')` to `canvas.getContext('webgl2-disabled-for-test')` in `matrix.js`, reload the page, and confirm the fallback message ("This experiment needs WebGL2…") renders instead of a blank page or console crash. Then revert the change back to `'webgl2'`.

- [ ] **Step 5: Commit**

```bash
git add new/experiments/matrix/index.html new/experiments/matrix/matrix.js
git commit -m "feat: add WebGL2 matrix rain shader experiment"
```

---

### Task 3: Audio-reactive visualizer experiment

**Files:**
- Create: `new/experiments/audio-visualizer/index.html`
- Create: `new/experiments/audio-visualizer/visualizer.js`

**Interfaces:**
- Consumes: `new/style.css` classes `.fallback-message`, `.back-link` from Task 1.

- [ ] **Step 1: Write `new/experiments/audio-visualizer/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex, nofollow">
<title>Audio Visualizer — Experiments</title>
<link rel="stylesheet" href="../../style.css">
<style>
  body { overflow: hidden; }
  canvas { display: block; }
  .audio-controls {
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 1rem;
    align-items: center;
    z-index: 10;
    background: rgba(10, 10, 15, 0.7);
    padding: 1rem 1.5rem;
    border-radius: 12px;
  }
  .audio-controls button,
  .audio-controls label {
    color: var(--fg);
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.9rem;
  }
  .audio-status { color: var(--muted); font-size: 0.85rem; }
  input[type="file"] { display: none; }
</style>
</head>
<body>
  <a class="back-link" href="../../index.html">← back</a>
  <div class="audio-controls">
    <button id="mic-button">Use microphone</button>
    <label for="file-input">Upload audio</label>
    <input type="file" id="file-input" accept="audio/*">
    <span class="audio-status" id="audio-status">Choose a source to start</span>
  </div>
  <script type="module" src="visualizer.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `new/experiments/audio-visualizer/visualizer.js`**

```javascript
import * as THREE from 'https://esm.sh/three@0.165.0';

const statusEl = document.getElementById('audio-status');
const micButton = document.getElementById('mic-button');
const fileInput = document.getElementById('file-input');

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 128;
const frequencyData = new Uint8Array(analyser.frequencyBinCount);

let sourceConnected = false;

function connectSource(sourceNode) {
  sourceNode.connect(analyser);
  sourceConnected = true;
}

async function startMic() {
  try {
    statusEl.textContent = 'Requesting microphone…';
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioCtx.createMediaStreamSource(stream);
    connectSource(source);
    statusEl.textContent = 'Listening to microphone';
  } catch (err) {
    statusEl.textContent = 'Microphone unavailable — upload a track instead';
    console.warn('Microphone access failed:', err);
  }
}

async function startFile(file) {
  try {
    statusEl.textContent = 'Decoding audio…';
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = true;
    connectSource(source);
    source.connect(audioCtx.destination);
    source.start();
    statusEl.textContent = `Playing "${file.name}"`;
  } catch (err) {
    statusEl.textContent = 'Could not read that audio file — try a different one';
    console.warn('Audio decode failed:', err);
  }
}

micButton.addEventListener('click', () => {
  audioCtx.resume();
  startMic();
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    audioCtx.resume();
    startFile(file);
  }
});

// --- Three.js scene ---

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 4, 14);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const barCount = analyser.frequencyBinCount;
const geometry = new THREE.BoxGeometry(0.3, 1, 0.3);
const material = new THREE.MeshBasicMaterial({ color: 0x7cfcae });
const bars = new THREE.InstancedMesh(geometry, material, barCount);
scene.add(bars);

const dummy = new THREE.Object3D();
const spacing = 0.4;
const startX = -((barCount - 1) * spacing) / 2;

function layoutBars(heights) {
  for (let i = 0; i < barCount; i++) {
    const h = Math.max(0.05, heights[i]);
    dummy.position.set(startX + i * spacing, h / 2, 0);
    dummy.scale.set(1, h, 1);
    dummy.updateMatrix();
    bars.setMatrixAt(i, dummy.matrix);
  }
  bars.instanceMatrix.needsUpdate = true;
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', resize);

function animate() {
  if (sourceConnected) {
    analyser.getByteFrequencyData(frequencyData);
    const heights = Array.from(frequencyData, (v) => (v / 255) * 8);
    layoutBars(heights);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
```

- [ ] **Step 3: Verify the microphone path locally**

Run: `python3 -m http.server 8000` from the repo root (if not already running).
Visit: `http://localhost:8000/new/experiments/audio-visualizer/index.html`
Click "Use microphone", allow the permission prompt, and speak/play sound near your mic.
Expected: status reads "Listening to microphone" and the green 3D bars react to volume/pitch in real time.

- [ ] **Step 4: Verify the mic-denied fallback**

Reload the page, click "Use microphone" again, and this time deny the browser's permission prompt.
Expected: status reads "Microphone unavailable — upload a track instead" (not stuck on "Requesting microphone…").

- [ ] **Step 5: Verify the file-upload path and the corrupt-file fallback**

Click "Upload audio" and choose a real audio file (mp3/wav). Expected: status reads `Playing "<filename>"` and bars react to the playback.
Then reload and try uploading a non-audio file (e.g. rename a `.txt` file to `.mp3` or pick an empty file). Expected: status reads "Could not read that audio file — try a different one" instead of hanging on "Decoding audio…".

- [ ] **Step 6: Commit**

```bash
git add new/experiments/audio-visualizer/index.html new/experiments/audio-visualizer/visualizer.js
git commit -m "feat: add Three.js audio-reactive visualizer experiment"
```

---

### Task 4: Hero scroll-driven animation

**Files:**
- Create: `new/js/hero.js`
- Modify: `new/index.html` (add script tag before `</body>`)

**Interfaces:**
- Consumes: `.hero`, `.hero-title`, `.hero-subtitle` classes from Task 1's `style.css`/`index.html`.

- [ ] **Step 1: Write `new/js/hero.js`**

```javascript
import { gsap } from 'https://esm.sh/gsap@3.12.5';
import { ScrollTrigger } from 'https://esm.sh/gsap@3.12.5/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const title = document.querySelector('.hero-title');
const subtitle = document.querySelector('.hero-subtitle');

if (title && subtitle) {
  gsap.set([title, subtitle], { opacity: 0, y: 40 });

  gsap
    .timeline({ defaults: { ease: 'power3.out' } })
    .to(title, { opacity: 1, y: 0, duration: 1 })
    .to(subtitle, { opacity: 1, y: 0, duration: 0.8 }, '-=0.5');

  gsap.to('.hero', {
    opacity: 0.2,
    scale: 0.9,
    scrollTrigger: {
      trigger: '.hero',
      start: 'top top',
      end: 'bottom top',
      scrub: true,
    },
  });
}
```

- [ ] **Step 2: Modify `new/index.html`** — add before `</body>`:

```html
  <script type="module" src="js/hero.js"></script>
```

- [ ] **Step 3: Verify locally**

Run: `python3 -m http.server 8000` from the repo root (if not already running).
Visit: `http://localhost:8000/new/`
Expected: on load, the title and subtitle fade/slide in; scrolling down fades and shrinks the hero section as it leaves the viewport.

- [ ] **Step 4: Commit**

```bash
git add new/js/hero.js new/index.html
git commit -m "feat: add scroll-driven hero animation with GSAP"
```

---

### Task 5: Tile grid reveal-on-scroll

**Files:**
- Create: `new/js/tiles.js`
- Modify: `new/index.html` (add script tag before `</body>`)

**Interfaces:**
- Consumes: `[data-tile]` attribute and `.tile`/`.tile.is-visible` classes from Task 1.

- [ ] **Step 1: Write `new/js/tiles.js`**

```javascript
const tiles = document.querySelectorAll('[data-tile]');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 }
);

tiles.forEach((tile) => observer.observe(tile));
```

- [ ] **Step 2: Modify `new/index.html`** — add before `</body>`, after the hero script:

```html
  <script type="module" src="js/tiles.js"></script>
```

- [ ] **Step 3: Verify locally**

Run: `python3 -m http.server 8000` from the repo root (if not already running).
Visit: `http://localhost:8000/new/`
Expected: scrolling down, each tile fades/slides in as it enters the viewport (instead of being visible immediately); clicking either tile navigates to its experiment page (both now exist from Tasks 2-3) and the "← back" link returns here.

- [ ] **Step 4: Commit**

```bash
git add new/js/tiles.js new/index.html
git commit -m "feat: add reveal-on-scroll behavior for experiment tiles"
```

---

### Task 6: Cross-browser and mobile verification pass

**Files:**
- Modify: any of the files above, only if a real bug is found during this pass (no changes expected by default).

- [ ] **Step 1: Full walkthrough in Chrome**

Run: `python3 -m http.server 8000` from the repo root.
In Chrome, visit `http://localhost:8000/new/`, scroll through the hero and grid, open both experiments, and exercise the mic-allow, mic-deny, and file-upload (valid + invalid file) paths from Task 3.
Expected: everything behaves as described in Tasks 1-5's verification steps, no console errors.

- [ ] **Step 2: Repeat the same walkthrough in Firefox and Safari**

Expected: same behavior as Chrome. Note any visual or functional differences.

- [ ] **Step 3: Verify narrow/mobile viewport**

In any browser's device toolbar (e.g. Chrome DevTools responsive mode), set the viewport to a phone width (~375px).
Expected: hero title/subtitle scale down and stay centered without horizontal scroll; tiles stack to a single column; the matrix canvas and audio visualizer canvas both fill the viewport without overflow; the audio-visualizer control bar remains tappable and doesn't overlap other content.

- [ ] **Step 4: Fix any issues found**

If any of the above steps surfaced a real bug (not a cosmetic nitpick), fix it in the relevant file from Tasks 1-5, and re-run that step's verification to confirm the fix.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix: address cross-browser/mobile issues found in verification pass"
```

(Skip this commit if Step 4 found nothing to fix.)
