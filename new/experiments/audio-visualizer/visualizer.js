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
