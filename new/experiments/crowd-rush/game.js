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

  // --- City blocks (solid obstacles) ---

  const buildings = []; // { minX, maxX, minZ, maxZ }
  const buildingColors = [0xf2d7a0, 0xa7c7e7, 0xf4a6a6, 0xb8e0c2];
  const gridPositions = [-24, -12, 0, 12, 24];

  gridPositions.forEach((bx) => {
    gridPositions.forEach((bz) => {
      if (bx === 0 && bz === 0) return; // keep the center plaza open as the start point
      const size = 8;
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
    });
  });

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

  const personGeometry = new THREE.ConeGeometry(0.28, 0.9, 6);

  function crowdRadius(count) {
    return 0.6 + Math.sqrt(count) * 0.35;
  }

  // Road gaps between buildings are 4 units wide. Capture range should keep
  // growing with crowd size, but the radius used to push the crowd's
  // *centroid* off buildings must stay under half that gap, or a big crowd
  // gets wedged between two buildings pushing it apart from both sides at
  // once. Capping it here lets a large crowd's edges visually spill past
  // building corners while its center still threads the road.
  function buildingCollisionRadius(count) {
    return Math.min(crowdRadius(count), 1.3);
  }

  function layoutFormation(mesh, count, centerX, centerZ) {
    const dummy = new THREE.Object3D();
    let placed = 0;
    let ring = 0;
    while (placed < count) {
      const ringCapacity = ring === 0 ? 1 : ring * 6;
      const ringRadius = ring * 0.55;
      for (let i = 0; i < ringCapacity && placed < count; i++) {
        const angle = (i / ringCapacity) * Math.PI * 2 + ring * 0.3;
        const x = centerX + Math.cos(angle) * ringRadius;
        const z = centerZ + Math.sin(angle) * ringRadius;
        dummy.position.set(x, 0.45, z);
        dummy.rotation.y = angle;
        dummy.updateMatrix();
        mesh.setMatrixAt(placed, dummy.matrix);
        placed++;
      }
      ring++;
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
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

  // --- Player ---

  const player = {
    position: new THREE.Vector3(0, 0, 0),
    count: 15,
    mesh: makeCrowdMesh(0x3a7bd5, PLAYER_CAP),
  };
  layoutFormation(player.mesh, player.count, player.position.x, player.position.z);

  // --- Rival crowds — every crowd on the map is an enemy: bigger absorbs
  // smaller on contact. Counts ascend with spawn index (with some jitter)
  // so early rivals are beatable from the player's starting size of 5 and
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
    // Capacity is generous headroom, not the expected size: rivals can now
    // absorb each other, so a single rival could in the worst case end up
    // holding close to the whole 10-rival pool.
    const mesh = makeCrowdMesh(rivalColors[i], 300);
    layoutFormation(mesh, count, pos.x, pos.z);
    rivalCrowds.push({
      name: rivalNames[i],
      position: pos,
      count,
      mesh,
      wanderTarget: pos.clone(),
      wanderTimer: 0,
    });
  }

  // --- Camera follow ---

  function updateCamera() {
    const behind = 14;
    const height = 16;
    camera.position.set(player.position.x, height, player.position.z + behind);
    camera.lookAt(player.position.x, 0, player.position.z - 4);
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

  // --- Game state / loop ---

  let timeLeft = ROUND_SECONDS;
  let gameOver = false;
  const clock = new THREE.Clock();

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
    moveToward(rival.position, rival.wanderTarget, 3.5, delta);
    resolveBuildingCollision(rival.position, buildingCollisionRadius(rival.count));
  }

  // Rivals also absorb each other on contact (bigger wins, same rule as
  // the player). Resolved one pair per pass and re-scanned from scratch
  // after each merge, since removing a rival shifts every index after it
  // — with at most 10 rivals this is cheap and avoids index-juggling bugs.
  function checkRivalVsRivalCollisions() {
    let resolvedAny = true;
    while (resolvedAny) {
      resolvedAny = false;
      for (let i = 0; i < rivalCrowds.length && !resolvedAny; i++) {
        for (let j = i + 1; j < rivalCrowds.length; j++) {
          const a = rivalCrowds[i];
          const b = rivalCrowds[j];
          if (a.count === b.count) continue;
          const dist = a.position.distanceTo(b.position);
          if (dist < crowdRadius(a.count) + crowdRadius(b.count)) {
            const winner = a.count > b.count ? a : b;
            const loser = a.count > b.count ? b : a;
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

  function checkCollisions() {
    for (let i = rivalCrowds.length - 1; i >= 0; i--) {
      const r = rivalCrowds[i];
      const dist = player.position.distanceTo(r.position);
      if (dist < crowdRadius(player.count) + crowdRadius(r.count)) {
        if (player.count > r.count) {
          player.count += r.count;
          scene.remove(r.mesh);
          rivalCrowds.splice(i, 1);
        } else if (r.count > player.count) {
          scene.remove(r.mesh);
          rivalCrowds.splice(i, 1);
          endRound('Defeated!');
          return;
        }
      }
    }

    if (rivalCrowds.length === 0) {
      endRound('All Rivals Defeated!');
    }
  }

  function animate() {
    const delta = Math.min(clock.getDelta(), 0.1);

    if (!gameOver) {
      timeLeft -= delta;
      if (timeLeft <= 0) {
        timeLeft = 0;
        endRound("Time's Up!");
      }

      const playerSpeed = 9;
      const keyDir = keyboardDirection();
      if (keyDir) {
        player.position.x += keyDir.x * playerSpeed * delta;
        player.position.z += keyDir.z * playerSpeed * delta;
      }
      resolveBuildingCollision(player.position, buildingCollisionRadius(player.count));

      rivalCrowds.forEach((r) => updateRivalAI(r, delta));
      checkRivalVsRivalCollisions();

      checkCollisions();

      layoutFormation(
        player.mesh,
        Math.min(player.count, PLAYER_CAP),
        player.position.x,
        player.position.z
      );
      rivalCrowds.forEach((r) => layoutFormation(r.mesh, r.count, r.position.x, r.position.z));

      hudTimer.textContent = formatTime(timeLeft);
      hudCount.textContent = String(player.count);
      updateLeaderboard();
      updateCamera();
    }

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  updateLeaderboard();
  animate();
}
