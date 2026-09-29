import * as THREE from 'https://esm.sh/three@0.165.0';

const statusEl = document.getElementById('audio-status');
const micButton = document.getElementById('mic-button');
const fileInput = document.getElementById('file-input');

const AudioContextClass = window.AudioContext || window.webkitAudioContext;

function canUseWebGL() {
  try {
    const testCanvas = document.createElement('canvas');
    return !!(testCanvas.getContext('webgl2') || testCanvas.getContext('webgl'));
  } catch (e) {
    return false;
  }
}

if (!AudioContextClass || !canUseWebGL()) {
  document.body.innerHTML =
    '<div class="fallback-message">This experiment needs Web Audio and WebGL, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
} else {
  const audioCtx = new AudioContextClass();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 128;
  const frequencyData = new Uint8Array(analyser.frequencyBinCount);

  let sourceConnected = false;
  let currentSource = null;
  let currentStream = null;

  function stopCurrentSource() {
    if (currentSource) {
      try {
        currentSource.disconnect();
      } catch (e) {
        // already disconnected
      }
      if (typeof currentSource.stop === 'function') {
        try {
          currentSource.stop();
        } catch (e) {
          // already stopped
        }
      }
    }
    if (currentStream) {
      currentStream.getTracks().forEach((track) => track.stop());
      currentStream = null;
    }
    currentSource = null;
  }

  function connectSource(sourceNode) {
    stopCurrentSource();
    sourceNode.connect(analyser);
    currentSource = sourceNode;
    sourceConnected = true;
  }

  async function startMic() {
    try {
      statusEl.textContent = 'Requesting microphone…';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioCtx.createMediaStreamSource(stream);
      connectSource(source);
      currentStream = stream;
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

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  const barCount = analyser.frequencyBinCount;
  const spacing = 0.4;
  const maxBarHeight = 8;
  const halfWidth = ((barCount - 1) * spacing) / 2;
  const halfHeight = maxBarHeight / 2;
  const framePadding = 1.15;

  const geometry = new THREE.BoxGeometry(0.3, 1, 0.3);
  const material = new THREE.MeshBasicMaterial({ color: 0x7cfcae });
  const bars = new THREE.InstancedMesh(geometry, material, barCount);
  scene.add(bars);

  const dummy = new THREE.Object3D();
  const startX = -halfWidth;

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

  // Frames the camera so the full bar array — at maximum possible bar
  // height, not just the current heights — stays in view at any aspect
  // ratio, instead of a fixed camera position that only fits a 3:2-ish
  // window and crops on mobile portrait or ultra-wide desktops.
  function updateCameraFraming() {
    const verticalFov = (camera.fov * Math.PI) / 180;
    const distanceForHeight = halfHeight / Math.tan(verticalFov / 2);
    const distanceForWidth = halfWidth / (Math.tan(verticalFov / 2) * camera.aspect);
    const distance = Math.max(distanceForHeight, distanceForWidth) * framePadding;
    camera.position.set(0, halfHeight * 0.6, distance);
    camera.lookAt(0, halfHeight * 0.4, 0);
    camera.updateProjectionMatrix();
  }

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    updateCameraFraming();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  window.addEventListener('resize', resize);

  // Draw a flat baseline immediately so the scene never renders as an
  // empty black screen while waiting for a source to connect.
  layoutBars(new Array(barCount).fill(0.05));
  updateCameraFraming();

  function animate() {
    if (sourceConnected) {
      analyser.getByteFrequencyData(frequencyData);
      const heights = Array.from(frequencyData, (v) => (v / 255) * maxBarHeight);
      layoutBars(heights);
    }
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
}
