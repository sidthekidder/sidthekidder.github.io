import * as THREE from 'https://esm.sh/three@0.165.0';

function canUseWebGL() {
  try {
    const testCanvas = document.createElement('canvas');
    return !!(testCanvas.getContext('webgl2') || testCanvas.getContext('webgl'));
  } catch (e) {
    return false;
  }
}

if (!canUseWebGL()) {
  document.body.innerHTML =
    '<a class="back-link" href="../../index.html">← back</a><div class="fallback-message">This experiment needs WebGL, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
} else {
  const ARENA_HALF = 30;
  const ROUND_SECONDS = 60;
  const PLAYER_CAP = 500;
  // Soldiers are batched into "clusters" rather than simulated one at a
  // time — each cluster steers/separates as a unit and renders GROUP_SIZE
  // separate capsules (small fixed offsets), so it still reads as
  // individuals while keeping the simulated entity count down.
  const GROUP_SIZE = 4;
  const GROUP_RADIUS = 0.32;

  const hudTimer = document.getElementById('hud-timer');
  const hudCount = document.getElementById('hud-count');
  const leaderboardEl = document.getElementById('leaderboard');
  const endScreen = document.getElementById('end-screen');
  const endTitle = document.getElementById('end-title');
  const endScore = document.getElementById('end-score');
  const playAgainBtn = document.getElementById('play-again');

  playAgainBtn.addEventListener('click', () => {
    window.location.reload();
  });

  // --- Sound ---
  // Synthesized with plain oscillators (no audio asset files, fitting the
  // rest of this page's no-build-step approach). Browsers require a user
  // gesture before an AudioContext can actually produce sound, so every
  // play function resumes it defensively — by the time any of these fire
  // the player has already pressed a movement key or touched the
  // joystick, but this covers the edge case cheaply either way.

  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  function ensureAudioResumed() {
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  function playTone(freqStart, freqEnd, duration, type, volume) {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    const now = audioCtx.currentTime;
    osc.frequency.setValueAtTime(freqStart, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(freqEnd, 1), now + duration);
    gain.gain.setValueAtTime(volume, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  function playAbsorbSound() {
    ensureAudioResumed();
    playTone(320, 720, 0.12, 'triangle', 0.15);
  }

  function playDefeatSound() {
    ensureAudioResumed();
    playTone(420, 110, 0.35, 'sawtooth', 0.2);
  }

  function playVictorySound() {
    ensureAudioResumed();
    const notes = [523.25, 659.25, 783.99]; // C5, E5, G5
    notes.forEach((freq, i) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      const startTime = audioCtx.currentTime + i * 0.1;
      gain.gain.setValueAtTime(0.18, startTime);
      gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.25);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(startTime);
      osc.stop(startTime + 0.25);
    });
  }

  // --- Scene setup ---

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xbfc9d6);

  const camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.1,
    200
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
  const sunLight = new THREE.DirectionalLight(0xffffff, 0.6);
  sunLight.position.set(20, 30, 10);
  scene.add(sunLight);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(ARENA_HALF * 2 + 20, ARENA_HALF * 2 + 20),
    new THREE.MeshLambertMaterial({ color: 0x9aa3ad })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // --- City blocks: a different mix of buildings/parks/open plazas every
  // time the page loads (Play Again reloads the page, so every round gets
  // a fresh layout for free). No physics engine involved: groups avoid
  // buildings with the same cheap AABB push-out used for the logical
  // leader position below; parks and open cells have no collision at all.

  const buildings = []; // { minX, maxX, minZ, maxZ } — buildings only
  const buildingColors = [0xf2d7a0, 0xa7c7e7, 0xf4a6a6, 0xb8e0c2];
  const gridPositions = [-24, -12, 0, 12, 24];

  function makeTree(x, z) {
    const trunkHeight = 0.6 + Math.random() * 0.3;
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.1, trunkHeight, 6),
      new THREE.MeshLambertMaterial({ color: 0x8b6b4a })
    );
    trunk.position.set(x, trunkHeight / 2, z);
    scene.add(trunk);

    const foliageHeight = 1.2 + Math.random() * 0.6;
    const foliage = new THREE.Mesh(
      new THREE.ConeGeometry(0.55 + Math.random() * 0.2, foliageHeight, 7),
      new THREE.MeshLambertMaterial({ color: 0x4a8f5c })
    );
    foliage.position.set(x, trunkHeight + foliageHeight / 2 - 0.05, z);
    scene.add(foliage);
  }

  function makeParkCell(bx, bz, size) {
    const patch = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshLambertMaterial({ color: 0x8fd18f })
    );
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(bx, 0.02, bz); // just above the road plane, avoids z-fighting
    scene.add(patch);

    const treeCount = 2 + Math.floor(Math.random() * 3);
    for (let t = 0; t < treeCount; t++) {
      const tx = bx + (Math.random() - 0.5) * (size - 1.5);
      const tz = bz + (Math.random() - 0.5) * (size - 1.5);
      makeTree(tx, tz);
    }
  }

  // Weighted so most cells are still buildings (keeps the "city" feel and
  // enough obstacles for navigation to matter), with parks and open
  // plazas mixed in for visual variety and breathing room.
  gridPositions.forEach((bx) => {
    gridPositions.forEach((bz) => {
      if (bx === 0 && bz === 0) return; // keep the center plaza open as the start point

      const roll = Math.random();
      if (roll < 0.55) {
        // Varied footprint size (not a fixed 8) so blocks don't all read
        // as identical cubes — a small step toward the "randomly combine
        // squares" building-size variation real city generators use.
        const size = 5 + Math.random() * 3;
        const height = 4 + Math.random() * 6;
        const color = buildingColors[Math.floor(Math.random() * buildingColors.length)];
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(size, height, size),
          new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.55 })
        );
        mesh.position.set(bx, height / 2, bz);
        scene.add(mesh);
        buildings.push({
          minX: bx - size / 2,
          maxX: bx + size / 2,
          minZ: bz - size / 2,
          maxZ: bz + size / 2,
        });
      } else if (roll < 0.78) {
        makeParkCell(bx, bz, 7);
      }
      // else (22%): open plaza — left empty, widens the road network there
    });
  });

  // Shared by the logical leader position's navigation AND every group's
  // per-frame building avoidance (see updateCrowdMotion below).
  function resolveBuildingCollision(position, radius) {
    for (const b of buildings) {
      const closestX = Math.max(b.minX, Math.min(position.x, b.maxX));
      const closestZ = Math.max(b.minZ, Math.min(position.z, b.maxZ));
      const dx = position.x - closestX;
      const dz = position.z - closestZ;
      const distSq = dx * dx + dz * dz;
      if (distSq < radius * radius) {
        const dist = Math.sqrt(distSq) || 0.0001;
        const push = radius - dist;
        position.x += (dx / dist) * push;
        position.z += (dz / dist) * push;
      }
    }
    position.x = Math.max(-ARENA_HALF, Math.min(ARENA_HALF, position.x));
    position.z = Math.max(-ARENA_HALF, Math.min(ARENA_HALF, position.z));
  }

  function randomRoadPosition(pushRadius) {
    const pos = new THREE.Vector3(
      (Math.random() * 2 - 1) * ARENA_HALF,
      0,
      (Math.random() * 2 - 1) * ARENA_HALF
    );
    resolveBuildingCollision(pos, pushRadius);
    return pos;
  }

  // --- Crowd helpers ---

  const personGeometry = new THREE.CapsuleGeometry(0.16, 0.35, 3, 6);
  const PERSON_HALF_HEIGHT = 0.35 / 2 + 0.16;

  // Small fixed offsets so a GROUP_SIZE cluster still reads as a few
  // separate people huddled together rather than one fat capsule.
  const GROUP_OFFSETS = [
    { x: 0, z: 0 },
    { x: 0.22, z: 0.1 },
    { x: -0.18, z: 0.15 },
    { x: 0.05, z: -0.2 },
  ];

  let elapsedTime = 0;

  function crowdRadius(count) {
    return 0.6 + Math.sqrt(count) * 0.35;
  }

  // Road gaps between buildings are 4 units wide. Capture range should keep
  // growing with crowd size, but the radius used to push the *logical*
  // leader position off buildings must stay under half that gap, or a big
  // crowd gets wedged between two buildings at once.
  function buildingCollisionRadius(count) {
    return Math.min(crowdRadius(count), 1.3);
  }

  // Where soldier `index` (within a crowd of any size) belongs, blending
  // between a gathered circle (moveBlend 0) and a wedge trailing behind
  // the facing direction (moveBlend 1). This is a target for steering,
  // not a direct position write — separation from neighbors still gets
  // the final say each frame.
  //
  // Writes into `out` instead of returning a fresh object: this runs once
  // per group every frame, and repeated per-frame allocation here (and in
  // crowdCentroid) was found to cause periodic GC pauses — the textbook
  // signature of frame-rate "jumping"/stutter rather than uniformly-low FPS.
  function computeSlotPosition(index, centerX, centerZ, facingAngle, moveBlend, out) {
    let ring = 0;
    let ringStart = 0;
    let ringCapacity = 1;
    while (index >= ringStart + ringCapacity) {
      ringStart += ringCapacity;
      ring++;
      ringCapacity = ring * 6;
    }
    const posInRing = index - ringStart;
    const baseAngle = (posInRing / ringCapacity) * Math.PI * 2;
    const seed = index * 12.9898;
    const wobble = Math.sin(elapsedTime * 2.4 + seed) * 0.07;

    const ringRadius = ring * 0.55;
    const circleX = centerX + Math.cos(baseAngle) * (ringRadius + wobble);
    const circleZ = centerZ + Math.sin(baseAngle) * (ringRadius + wobble);

    const forwardX = Math.sin(facingAngle);
    const forwardZ = Math.cos(facingAngle);
    const rightX = Math.cos(facingAngle);
    const rightZ = -Math.sin(facingAngle);
    const depthOffset = ring * 0.42 + Math.cos(baseAngle) * (ring * 0.12) + wobble;
    const lateralOffset = Math.sin(baseAngle) * (ring * 0.5) + wobble;
    const trailX = centerX - forwardX * depthOffset + rightX * lateralOffset;
    const trailZ = centerZ - forwardZ * depthOffset + rightZ * lateralOffset;

    out.x = circleX + (trailX - circleX) * moveBlend;
    out.z = circleZ + (trailZ - circleZ) * moveBlend;
    return out;
  }

  // The crowd's real visual center — the size-weighted average of where
  // its groups actually are, not the abstract WASD-driven intent point
  // (crowd.position). Used for anything that should reflect what's
  // actually on screen: collision/contact checks and the camera.
  function crowdCentroid(crowd, out) {
    let sumX = 0;
    let sumZ = 0;
    let totalSize = 0;
    crowd.groups.forEach((group) => {
      sumX += group.position.x * group.size;
      sumZ += group.position.z * group.size;
      totalSize += group.size;
    });
    const n = totalSize || 1;
    const target = out || { x: 0, z: 0 };
    target.x = sumX / n;
    target.z = sumZ / n;
    return target;
  }

  function makeCrowdMesh(color, capacity) {
    const material = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.InstancedMesh(personGeometry, material, capacity);
    // Instances are placed via per-instance matrices at their world
    // position while the mesh itself never moves from local origin, so
    // Three.js's default frustum-culling bounds (computed from the base
    // geometry at that untouched origin) don't cover where the crowd
    // actually is. Without this, the whole mesh gets culled as soon as
    // the crowd wanders far enough from world origin.
    mesh.frustumCulled = false;
    scene.add(mesh);
    return mesh;
  }

  // Spawns GROUP_SIZE-soldier clusters already at their formation slot.
  // Each group's target slot is computed at its first individual index —
  // close enough to where the rest of that cluster would sit, since a few
  // consecutive formation indices are always spatially near each other.
  function spawnGroups(count, x, z) {
    const groups = [];
    let placed = 0;
    while (placed < count) {
      const size = Math.min(GROUP_SIZE, count - placed);
      const slot = computeSlotPosition(placed, x, z, 0, 0, { x: 0, z: 0 });
      groups.push({
        position: { x: slot.x, z: slot.z },
        velocity: { x: 0, z: 0 },
        size,
      });
      placed += size;
    }
    return groups;
  }

  // --- Player ---

  const player = {
    position: new THREE.Vector3(0, 0, 0),
    count: 15,
    facing: 0,
    moveBlend: 0,
    mesh: makeCrowdMesh(0x3a7bd5, PLAYER_CAP),
    groups: [],
  };
  player.groups = spawnGroups(player.count, player.position.x, player.position.z);

  // --- Rival crowds — every crowd on the map is an enemy: bigger absorbs
  // smaller on contact. Counts ascend with spawn index (with some jitter)
  // so early rivals are beatable from the player's starting size of 15 and
  // later ones demand you've grown first. ---

  const rivalColors = [
    0xe15554, 0xff8c42, 0x9b5de5, 0xf9c74f, 0x43aa8b,
    0xf72585, 0x577590, 0x90be6d, 0xf3722c, 0x4d908e,
  ];
  const rivalNames = [
    'Redcoat', 'Blazer', 'Violet', 'Marigold', 'Ember',
    'Magenta', 'Slate', 'Clover', 'Rust', 'Teal',
  ];
  const RIVAL_COUNT = 10;
  const rivalCrowds = [];

  for (let i = 0; i < RIVAL_COUNT; i++) {
    const count = 2 + i * 3 + Math.floor(Math.random() * 4);
    const pos = randomRoadPosition(buildingCollisionRadius(count));
    const mesh = makeCrowdMesh(rivalColors[i], 300);
    const groups = spawnGroups(count, pos.x, pos.z);
    rivalCrowds.push({
      name: rivalNames[i],
      color: rivalColors[i],
      position: pos,
      count,
      facing: 0,
      moveBlend: 0,
      mesh,
      groups,
      wanderTarget: pos.clone(),
      wanderTimer: 0,
    });
  }

  // --- Camera follow ---

  // Follows the crowd's actual visual centroid, not the logical
  // WASD-driven position — the logical point moves at a flat speed with
  // no resistance, so it would otherwise steadily pull ahead of the real
  // (steered, separated) crowd and leave the camera looking at empty
  // space in front of the group. Using the centroid of every real group
  // (rather than just one) also means a stuck straggler can't drag the
  // camera off if the rest of the crowd has moved on.
  const cameraCentroidScratch = { x: 0, z: 0 };

  // Dollies out as the crowd grows (based on the same crowdRadius used for
  // capture range), eased over time rather than snapped, so a big absorb
  // doesn't yank the camera back in one frame. `delta` is optional so the
  // very first call (before the render loop starts) can snap straight to
  // the correct starting zoom instead of easing in from 1x.
  let cameraZoom = 1;

  // `target` defaults to the player — but during the defeat sequence
  // (see pendingDefeat in animate()) the camera follows whichever rival
  // just absorbed the player instead, since player.groups is empty by
  // then and would otherwise centroid to nothing.
  function updateCamera(target, delta) {
    const crowd = target || player;
    const baseBehind = 14;
    const baseHeight = 16;
    const centroid = crowdCentroid(crowd, cameraCentroidScratch);

    const targetZoom = Math.min(
      2,
      1 + Math.max(0, crowdRadius(crowd.count) - crowdRadius(15)) * 0.12
    );
    const zoomEase = Math.min(1, (delta === undefined ? 1 : delta) * 2);
    cameraZoom += (targetZoom - cameraZoom) * zoomEase;

    camera.position.set(centroid.x, baseHeight * cameraZoom, centroid.z + baseBehind * cameraZoom);
    camera.lookAt(centroid.x, 0, centroid.z - 4);
  }
  updateCamera();

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  window.addEventListener('resize', resize);

  // --- Keyboard input: WASD / arrow keys are the only movement control ---

  const keys = { up: false, down: false, left: false, right: false };
  const movementKeyCodes = new Set([
    'KeyW',
    'KeyA',
    'KeyS',
    'KeyD',
    'ArrowUp',
    'ArrowDown',
    'ArrowLeft',
    'ArrowRight',
  ]);

  function setKeyState(code, value) {
    switch (code) {
      case 'KeyW':
      case 'ArrowUp':
        keys.up = value;
        break;
      case 'KeyS':
      case 'ArrowDown':
        keys.down = value;
        break;
      case 'KeyA':
      case 'ArrowLeft':
        keys.left = value;
        break;
      case 'KeyD':
      case 'ArrowRight':
        keys.right = value;
        break;
    }
  }

  window.addEventListener('keydown', (e) => {
    if (movementKeyCodes.has(e.code)) e.preventDefault();
    setKeyState(e.code, true);
  });
  window.addEventListener('keyup', (e) => setKeyState(e.code, false));

  function keyboardDirection() {
    let dx = 0;
    let dz = 0;
    if (keys.up) dz -= 1;
    if (keys.down) dz += 1;
    if (keys.left) dx -= 1;
    if (keys.right) dx += 1;
    if (dx === 0 && dz === 0) return null;
    const len = Math.sqrt(dx * dx + dz * dz);
    return { x: dx / len, z: dz / len };
  }

  // --- Touch input: on-screen virtual joystick for mobile ---
  // Separate from (and not a replacement for) the whole-screen
  // pointer-follow control removed earlier — this only responds to drags
  // starting on the joystick element itself, and is hidden entirely on
  // non-touch (fine) pointers via CSS.

  const joystickEl = document.getElementById('joystick');
  const joystickKnobEl = document.getElementById('joystick-knob');
  const JOYSTICK_MAX_RADIUS = 45;
  const JOYSTICK_DEADZONE = 8;

  let joystickPointerId = null;
  let joystickDirX = 0;
  let joystickDirZ = 0;

  function updateJoystickFromEvent(clientX, clientY) {
    const rect = joystickEl.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > JOYSTICK_MAX_RADIUS) {
      dx = (dx / dist) * JOYSTICK_MAX_RADIUS;
      dy = (dy / dist) * JOYSTICK_MAX_RADIUS;
    }
    joystickKnobEl.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    if (Math.sqrt(dx * dx + dy * dy) < JOYSTICK_DEADZONE) {
      joystickDirX = 0;
      joystickDirZ = 0;
    } else {
      // dy grows downward on screen, matching how W/ArrowUp already map to
      // negative dz ("forward" for this fixed camera) — no sign flip needed.
      joystickDirX = dx / JOYSTICK_MAX_RADIUS;
      joystickDirZ = dy / JOYSTICK_MAX_RADIUS;
    }
  }

  function resetJoystick() {
    joystickPointerId = null;
    joystickDirX = 0;
    joystickDirZ = 0;
    joystickKnobEl.style.transform = 'translate(-50%, -50%)';
  }

  joystickEl.addEventListener('pointerdown', (e) => {
    joystickPointerId = e.pointerId;
    joystickEl.setPointerCapture(e.pointerId);
    updateJoystickFromEvent(e.clientX, e.clientY);
  });
  joystickEl.addEventListener('pointermove', (e) => {
    if (e.pointerId === joystickPointerId) {
      updateJoystickFromEvent(e.clientX, e.clientY);
    }
  });
  joystickEl.addEventListener('pointerup', (e) => {
    if (e.pointerId === joystickPointerId) resetJoystick();
  });
  joystickEl.addEventListener('pointercancel', (e) => {
    if (e.pointerId === joystickPointerId) resetJoystick();
  });
  window.addEventListener('blur', resetJoystick);

  function joystickDirection() {
    if (joystickPointerId === null) return null;
    const len = Math.sqrt(joystickDirX * joystickDirX + joystickDirZ * joystickDirZ);
    if (len < 0.001) return null;
    return { x: joystickDirX / len, z: joystickDirZ / len };
  }

  // --- Game state / loop ---

  let timeLeft = ROUND_SECONDS;
  let gameOver = false;
  const clock = new THREE.Clock();

  // When the player loses, instead of freezing instantly: the player's
  // groups transfer into the winning rival (visually "converting" into
  // its crowd/color) and the camera follows that rival for DEFEAT_DELAY
  // seconds before the end screen actually shows.
  const DEFEAT_DELAY = 1;
  let pendingDefeat = null; // { rival, timer } while the sequence plays out

  function formatTime(seconds) {
    const s = Math.max(0, Math.ceil(seconds));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}:${rem.toString().padStart(2, '0')}`;
  }

  function endRound(title) {
    if (gameOver) return;
    gameOver = true;
    endTitle.textContent = title;
    endScore.textContent = `Final size: ${player.count}`;
    endScreen.classList.add('is-visible');
    if (title === 'Defeated!') {
      playDefeatSound();
    } else {
      playVictorySound();
    }
  }

  function updateLeaderboard() {
    const entries = [{ name: 'You', count: player.count, you: true }].concat(
      rivalCrowds.map((r) => ({ name: r.name, count: r.count, you: false }))
    );
    entries.sort((a, b) => b.count - a.count);
    leaderboardEl.innerHTML = entries
      .map(
        (e) =>
          `<div class="${e.you ? 'you' : ''}"><span>${e.name}</span><span>${e.count}</span></div>`
      )
      .join('');
  }

  function moveToward(position, target, speed, delta) {
    const dx = target.x - position.x;
    const dz = target.z - position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist < 0.05) return;
    const step = Math.min(dist, speed * delta);
    position.x += (dx / dist) * step;
    position.z += (dz / dist) * step;
  }

  function updateRivalAI(rival, delta) {
    rival.wanderTimer -= delta;
    if (rival.wanderTimer <= 0) {
      rival.wanderTarget = randomRoadPosition(buildingCollisionRadius(rival.count));
      rival.wanderTimer = 2 + Math.random() * 3;
    }
    const dx = rival.wanderTarget.x - rival.position.x;
    const dz = rival.wanderTarget.z - rival.position.z;
    const isMoving = dx * dx + dz * dz > 0.0025;
    if (isMoving) {
      rival.facing = Math.atan2(dx, dz);
    }
    const blendRate = Math.min(1, delta * 4);
    rival.moveBlend += ((isMoving ? 1 : 0) - rival.moveBlend) * blendRate;
    moveToward(rival.position, rival.wanderTarget, 3.5, delta);
    resolveBuildingCollision(rival.position, buildingCollisionRadius(rival.count));
  }

  // Rivals also absorb each other on contact (bigger wins, same rule as
  // the player) — the loser's groups transfer straight into the winner's
  // array (no destroy/recreate — they're plain state, not physics
  // bodies), so the absorbed soldiers keep their exact position/velocity
  // and visibly run to join the winner's new formation on later frames.
  function checkRivalVsRivalCollisions() {
    let resolvedAny = true;
    while (resolvedAny) {
      resolvedAny = false;
      // Each rival's centroid is the same for every pair it's checked
      // against within this pass — compute it once per rival here rather
      // than recomputing it from scratch on every pair comparison below.
      const centroids = rivalCrowds.map((r) => crowdCentroid(r));
      for (let i = 0; i < rivalCrowds.length && !resolvedAny; i++) {
        for (let j = i + 1; j < rivalCrowds.length; j++) {
          const a = rivalCrowds[i];
          const b = rivalCrowds[j];
          const dist = Math.hypot(
            centroids[i].x - centroids[j].x,
            centroids[i].z - centroids[j].z
          );
          if (dist < crowdRadius(a.count) + crowdRadius(b.count)) {
            const winner = a.count >= b.count ? a : b;
            const loser = a.count >= b.count ? b : a;
            const loserCentroid = a.count >= b.count ? centroids[j] : centroids[i];
            spawnBurst(loserCentroid.x, loserCentroid.z, loser.color);
            winner.groups = winner.groups.concat(loser.groups);
            winner.count += loser.count;
            scene.remove(loser.mesh);
            rivalCrowds.splice(rivalCrowds.indexOf(loser), 1);
            resolvedAny = true;
            break;
          }
        }
      }
    }
  }

  const playerCollisionCentroidScratch = { x: 0, z: 0 };
  const rivalCollisionCentroidScratch = { x: 0, z: 0 };

  function checkCollisions() {
    const playerCentroid = crowdCentroid(player, playerCollisionCentroidScratch);
    for (let i = rivalCrowds.length - 1; i >= 0; i--) {
      const r = rivalCrowds[i];
      const rivalCentroid = crowdCentroid(r, rivalCollisionCentroidScratch);
      const dist = Math.hypot(
        playerCentroid.x - rivalCentroid.x,
        playerCentroid.z - rivalCentroid.z
      );
      if (dist < crowdRadius(player.count) + crowdRadius(r.count)) {
        if (player.count >= r.count) {
          spawnBurst(rivalCentroid.x, rivalCentroid.z, r.color);
          playAbsorbSound();
          player.groups = player.groups.concat(r.groups);
          player.count += r.count;
          scene.remove(r.mesh);
          rivalCrowds.splice(i, 1);
        } else {
          spawnBurst(playerCentroid.x, playerCentroid.z, 0xff4d4d);
          // Player's groups visually convert into the winning rival's
          // crowd instead of just vanishing — r keeps existing (not
          // removed/spliced) so it can keep rendering them merging in
          // during the DEFEAT_DELAY window below. player.count is left
          // untouched: endRound's "Final size" reads it, and zeroing it
          // here would show 0 instead of the size you actually reached.
          const absorbedGroups = player.groups;
          r.groups = r.groups.concat(absorbedGroups);
          r.count += player.count;
          player.groups = [];
          pendingDefeat = { rival: r, timer: DEFEAT_DELAY };
          return;
        }
      }
    }

    if (rivalCrowds.length === 0) {
      endRound('All Rivals Defeated!');
    }
  }

  // --- Spatial hash ---
  // Replaces a full physics engine's broadphase for "which groups are
  // near this one" queries. Rebuilt fresh every frame (cheap — O(number
  // of groups)) rather than incrementally maintained, since groups move
  // every frame anyway. CELL_SIZE is a little larger than the separation
  // radius so a 3x3 cell neighborhood always covers it.

  const CELL_SIZE = 1.2;

  function cellKey(cx, cz) {
    return cx + ':' + cz;
  }

  function buildSpatialHash(allGroups) {
    const hash = new Map();
    for (const group of allGroups) {
      const cx = Math.floor(group.position.x / CELL_SIZE);
      const cz = Math.floor(group.position.z / CELL_SIZE);
      const key = cellKey(cx, cz);
      let bucket = hash.get(key);
      if (!bucket) {
        bucket = [];
        hash.set(key, bucket);
      }
      bucket.push(group);
    }
    return hash;
  }

  function forEachNearby(hash, x, z, callback) {
    const cx = Math.floor(x / CELL_SIZE);
    const cz = Math.floor(z / CELL_SIZE);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dz = -1; dz <= 1; dz++) {
        const bucket = hash.get(cellKey(cx + dx, cz + dz));
        if (bucket) {
          for (const other of bucket) callback(other);
        }
      }
    }
  }

  // --- Boids-style separation ---
  // This is what gives jostle/knockback now: groups (regardless of which
  // crowd owns them, same as the removed physics engine treated every
  // soldier) push apart from close neighbors. No rigid-body solver, no
  // gravity, no broadphase/narrowphase pipeline — just a spatial-hash
  // neighbor query and a direct force, which is the standard approach
  // real crowd/flocking simulations use instead of a general physics
  // engine (whose collision pipeline doesn't scale for "many simple
  // agents" the way it does for a handful of "real" rigid objects).

  const SEPARATION_RADIUS = 0.9;
  const SEPARATION_STRENGTH = 5;
  const separationScratch = { x: 0, z: 0 };

  function computeSeparation(group, hash, out) {
    out.x = 0;
    out.z = 0;
    forEachNearby(hash, group.position.x, group.position.z, (other) => {
      if (other === group) return;
      const dx = group.position.x - other.position.x;
      const dz = group.position.z - other.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz) || 0.001;
      if (dist < SEPARATION_RADIUS) {
        const push = (SEPARATION_RADIUS - dist) / SEPARATION_RADIUS;
        out.x += (dx / dist) * push;
        out.z += (dz / dist) * push;
      }
    });
    out.x *= SEPARATION_STRENGTH;
    out.z *= SEPARATION_STRENGTH;
    return out;
  }

  const slotScratch = { x: 0, z: 0 };

  // Base steering speed once within CATCHUP_START of the target slot.
  // Beyond that, speed ramps up with distance (capped at CATCHUP_MAX_SPEED)
  // so a group that's fallen behind sprints back rather than crawling at
  // the same speed as everyone else already in formation.
  const BASE_STEER_SPEED = 6;
  const CATCHUP_START = 3;
  const CATCHUP_RATE = 2.5;
  const CATCHUP_MAX_SPEED = 14;

  function steerSpeedForDistance(dist) {
    if (dist <= CATCHUP_START) return BASE_STEER_SPEED;
    return Math.min(
      CATCHUP_MAX_SPEED,
      BASE_STEER_SPEED + (dist - CATCHUP_START) * CATCHUP_RATE
    );
  }

  // Full per-frame motion update for one crowd: steer each group toward
  // its formation slot, separate from nearby groups (any crowd), avoid
  // buildings, then integrate position directly — no physics step
  // involved, this IS the step.
  function updateCrowdMotion(crowd, hash, delta) {
    let individualIndex = 0;
    crowd.groups.forEach((group) => {
      const slot = computeSlotPosition(
        individualIndex,
        crowd.position.x,
        crowd.position.z,
        crowd.facing,
        crowd.moveBlend,
        slotScratch
      );
      const dx = slot.x - group.position.x;
      const dz = slot.z - group.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);

      // No teleport/snap fallback: however far behind a group has fallen,
      // it always moves there continuously — steerSpeedForDistance's
      // catch-up ramp (capped at CATCHUP_MAX_SPEED) is the only recovery
      // mechanism now. A group can in principle stay stuck longer against
      // an awkward building corner than it could before, but it will
      // never visibly pop from one position to another.
      let steerX = 0;
      let steerZ = 0;
      if (dist > 0.02) {
        const speed = Math.min(steerSpeedForDistance(dist), dist * 8);
        steerX = (dx / dist) * speed;
        steerZ = (dz / dist) * speed;
      }
      const sep = computeSeparation(group, hash, separationScratch);
      group.velocity.x = steerX + sep.x;
      group.velocity.z = steerZ + sep.z;
      group.position.x += group.velocity.x * delta;
      group.position.z += group.velocity.z * delta;
      resolveBuildingCollision(group.position, GROUP_RADIUS);
      individualIndex += group.size;
    });
  }

  // --- Contact bursts ---
  // A quick expanding, fading ring dropped at the contact point on every
  // absorption (player or rival-vs-rival), tinted with whichever crowd
  // just got absorbed. Each burst is a short-lived Three.js mesh tracked
  // in activeBursts and cleaned up (removed + disposed) once its
  // animation finishes, so nothing accumulates over a round.

  const activeBursts = [];
  const BURST_DURATION = 0.45;
  const BURST_MAX_SCALE = 4;

  function spawnBurst(x, z, color) {
    const geometry = new THREE.RingGeometry(0.3, 0.5, 20);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, 0.4, z);
    scene.add(mesh);
    activeBursts.push({ mesh, material, age: 0 });
  }

  function updateBursts(delta) {
    for (let i = activeBursts.length - 1; i >= 0; i--) {
      const burst = activeBursts[i];
      burst.age += delta;
      const t = Math.min(1, burst.age / BURST_DURATION);
      const scale = 1 + t * (BURST_MAX_SCALE - 1);
      burst.mesh.scale.set(scale, scale, scale);
      burst.material.opacity = 0.9 * (1 - t);
      if (t >= 1) {
        scene.remove(burst.mesh);
        burst.mesh.geometry.dispose();
        burst.material.dispose();
        activeBursts.splice(i, 1);
      }
    }
  }

  const dummy = new THREE.Object3D();

  // Writes GROUP_SIZE instances per group into the crowd's InstancedMesh
  // — small fixed offsets around the group's position plus a per-instance
  // cosmetic bob/lean, so a cluster still reads as a few separate people.
  function renderCrowd(crowd) {
    let renderIndex = 0;
    crowd.groups.forEach((group) => {
      for (let k = 0; k < group.size; k++) {
        const offset = GROUP_OFFSETS[k];
        const seed = renderIndex * 12.9898;
        const bob = Math.abs(Math.sin(elapsedTime * 7 + seed)) * 0.14;
        const lean = Math.sin(elapsedTime * 5 + seed) * 0.12;
        dummy.position.set(
          group.position.x + offset.x,
          PERSON_HALF_HEIGHT + bob,
          group.position.z + offset.z
        );
        dummy.rotation.y = crowd.facing + lean;
        dummy.updateMatrix();
        crowd.mesh.setMatrixAt(renderIndex, dummy.matrix);
        renderIndex++;
      }
    });
    crowd.mesh.count = renderIndex;
    crowd.mesh.instanceMatrix.needsUpdate = true;
  }

  function animate() {
    const delta = Math.min(clock.getDelta(), 0.1);
    elapsedTime += delta;

    if (pendingDefeat) {
      // The 1-second "your crowd converts into the winner" beat: normal
      // gameplay (input, rival AI, timer, collision checks) is paused —
      // only motion/rendering keeps running so the just-absorbed player
      // groups visibly merge into pendingDefeat.rival's formation, with
      // the camera following that rival instead of the now-empty player.
      pendingDefeat.timer -= delta;

      let allGroups = player.groups;
      rivalCrowds.forEach((r) => {
        allGroups = allGroups.concat(r.groups);
      });
      const hash = buildSpatialHash(allGroups);

      updateCrowdMotion(player, hash, delta); // no-op: player.groups is empty
      rivalCrowds.forEach((r) => updateCrowdMotion(r, hash, delta));

      renderCrowd(player);
      rivalCrowds.forEach((r) => renderCrowd(r));

      updateLeaderboard();
      updateCamera(pendingDefeat.rival, delta);

      if (pendingDefeat.timer <= 0) {
        pendingDefeat = null;
        endRound('Defeated!');
      }
    } else if (!gameOver) {
      timeLeft -= delta;
      if (timeLeft <= 0) {
        timeLeft = 0;
        endRound("Time's Up!");
      }

      const playerSpeed = 6.5;
      const keyDir = keyboardDirection() || joystickDirection();
      const playerBlendRate = Math.min(1, delta * 4);
      player.moveBlend += ((keyDir ? 1 : 0) - player.moveBlend) * playerBlendRate;
      if (keyDir) {
        player.position.x += keyDir.x * playerSpeed * delta;
        player.position.z += keyDir.z * playerSpeed * delta;
        player.facing = Math.atan2(keyDir.x, keyDir.z);
      }
      resolveBuildingCollision(player.position, buildingCollisionRadius(player.count));

      rivalCrowds.forEach((r) => updateRivalAI(r, delta));
      checkRivalVsRivalCollisions();

      checkCollisions();

      // checkCollisions may have just set pendingDefeat (and cleared
      // player.groups) as a side effect — skip the rest of this frame's
      // normal-path rendering/camera in that case so updateCamera doesn't
      // centroid an empty player crowd to world origin for one frame
      // before the pendingDefeat branch takes over next frame.
      if (!pendingDefeat) {
        let allGroups = player.groups;
        rivalCrowds.forEach((r) => {
          allGroups = allGroups.concat(r.groups);
        });
        const hash = buildSpatialHash(allGroups);

        updateCrowdMotion(player, hash, delta);
        rivalCrowds.forEach((r) => updateCrowdMotion(r, hash, delta));

        renderCrowd(player);
        rivalCrowds.forEach((r) => renderCrowd(r));

        hudTimer.textContent = formatTime(timeLeft);
        hudCount.textContent = String(player.count);
        updateLeaderboard();
        updateCamera(player, delta);
      }
    }

    // Outside the !gameOver gate so a burst spawned on the frame the
    // round ends (a defeat or a final absorb) still finishes its
    // animation instead of freezing mid-fade.
    updateBursts(delta);

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  updateLeaderboard();
  animate();
}
