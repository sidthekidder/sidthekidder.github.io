import * as THREE from 'https://esm.sh/three@0.165.0';
import * as CANNON from 'https://esm.sh/cannon-es@0.20.0';

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
  const SOLDIER_RADIUS = 0.18;

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

  // --- Physics world ---
  // Every individual soldier is a real rigid body here, not just each
  // crowd's centroid — so soldiers physically jostle each other, collide
  // with buildings, and knock into rival crowds on contact. This is a
  // deliberate performance tradeoff (up to ~200 bodies at once) accepted
  // over the safer/cheaper "one body per crowd" approach, and hasn't been
  // verified against real frame-rate in a browser.

  const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.82, 0) });

  const groundBody = new CANNON.Body({ mass: 0, shape: new CANNON.Plane() });
  groundBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
  world.addBody(groundBody);

  // --- City blocks (solid obstacles — real physics bodies now, soldiers
  // collide with them directly instead of being algebraically pushed out) ---

  const buildings = []; // { minX, maxX, minZ, maxZ } — still used by the
  // logical leader position's own navigation, see resolveBuildingCollision
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

      const buildingBody = new CANNON.Body({
        mass: 0,
        shape: new CANNON.Box(new CANNON.Vec3(size / 2, 6, size / 2)),
        position: new CANNON.Vec3(bx, 6, bz),
      });
      world.addBody(buildingBody);
    });
  });

  // Still used for the crowd's *logical* leader position (win/lose checks,
  // camera target) — kept independent of the physics simulation so core
  // gameplay stays deterministic even under heavy physics load.
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
  // Aligns the capsule's visual base with where its physics sphere
  // actually rests on the ground (sphere settles at y = its own radius).
  const RENDER_Y_OFFSET = PERSON_HALF_HEIGHT - SOLDIER_RADIUS;

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
  // the facing direction (moveBlend 1). This is a target for physics
  // *steering*, not a direct position write — the solver has the final
  // say once collisions with neighbors/buildings are factored in.
  function computeSlotPosition(index, centerX, centerZ, facingAngle, moveBlend) {
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

    return {
      x: circleX + (trailX - circleX) * moveBlend,
      z: circleZ + (trailZ - circleZ) * moveBlend,
    };
  }

  function makeSoldierBody(x, z) {
    const body = new CANNON.Body({
      mass: 1,
      shape: new CANNON.Sphere(SOLDIER_RADIUS),
      position: new CANNON.Vec3(x, 0.5 + Math.random() * 0.5, z),
      linearDamping: 0.85,
      fixedRotation: true,
    });
    world.addBody(body);
    return body;
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

  // Spawns each soldier already near its formation slot (rather than all
  // stacked at one point), so the physics solver doesn't have to violently
  // separate a pile of exactly-overlapping bodies on the first step.
  function spawnBodies(count, x, z) {
    const bodies = [];
    for (let i = 0; i < count; i++) {
      const slot = computeSlotPosition(i, x, z, 0, 0);
      bodies.push(makeSoldierBody(slot.x, slot.z));
    }
    return bodies;
  }

  // --- Player ---

  const player = {
    position: new THREE.Vector3(0, 0, 0),
    count: 15,
    facing: 0,
    moveBlend: 0,
    mesh: makeCrowdMesh(0x3a7bd5, PLAYER_CAP),
    bodies: [],
  };
  player.bodies = spawnBodies(player.count, player.position.x, player.position.z);

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
    // Capacity is generous headroom, not the expected size: rivals can now
    // absorb each other, so a single rival could in the worst case end up
    // holding close to the whole 10-rival pool.
    const mesh = makeCrowdMesh(rivalColors[i], 300);
    const bodies = spawnBodies(count, pos.x, pos.z);
    rivalCrowds.push({
      name: rivalNames[i],
      position: pos,
      count,
      facing: 0,
      moveBlend: 0,
      mesh,
      bodies,
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
  // the player) — the loser's physics bodies transfer straight into the
  // winner's array (no need to destroy/recreate them), so the absorbed
  // soldiers keep their exact physics state and visibly run to join the
  // winner's new formation slots on the following frames.
  function checkRivalVsRivalCollisions() {
    let resolvedAny = true;
    while (resolvedAny) {
      resolvedAny = false;
      for (let i = 0; i < rivalCrowds.length && !resolvedAny; i++) {
        for (let j = i + 1; j < rivalCrowds.length; j++) {
          const a = rivalCrowds[i];
          const b = rivalCrowds[j];
          const dist = a.position.distanceTo(b.position);
          if (dist < crowdRadius(a.count) + crowdRadius(b.count)) {
            const winner = a.count >= b.count ? a : b;
            const loser = a.count >= b.count ? b : a;
            winner.bodies = winner.bodies.concat(loser.bodies);
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
        if (player.count >= r.count) {
          player.bodies = player.bodies.concat(r.bodies);
          player.count += r.count;
          scene.remove(r.mesh);
          rivalCrowds.splice(i, 1);
        } else {
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

  const dummy = new THREE.Object3D();

  // Sets each soldier's steering intent (velocity toward its formation
  // slot) before the physics step — the solver has the final say on
  // actual motion once collisions with neighbors/buildings are resolved.
  function steerCrowd(crowd) {
    const steerSpeed = 6;
    crowd.bodies.forEach((body, i) => {
      const slot = computeSlotPosition(
        i,
        crowd.position.x,
        crowd.position.z,
        crowd.facing,
        crowd.moveBlend
      );
      const dx = slot.x - body.position.x;
      const dz = slot.z - body.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist > 0.02) {
        const speed = Math.min(steerSpeed, dist * 8);
        body.velocity.x = (dx / dist) * speed;
        body.velocity.z = (dz / dist) * speed;
      } else {
        body.velocity.x *= 0.5;
        body.velocity.z *= 0.5;
      }
    });
  }

  // Reads back each soldier's resolved physics position (after the world
  // step) and writes it into the crowd's InstancedMesh, with a small
  // cosmetic bob/lean layered on top.
  function renderCrowd(crowd) {
    crowd.bodies.forEach((body, i) => {
      const seed = i * 12.9898;
      const bob = Math.abs(Math.sin(elapsedTime * 7 + seed)) * 0.14;
      const lean = Math.sin(elapsedTime * 5 + seed) * 0.12;
      dummy.position.set(
        body.position.x,
        body.position.y + RENDER_Y_OFFSET + bob,
        body.position.z
      );
      dummy.rotation.y = crowd.facing + lean;
      dummy.updateMatrix();
      crowd.mesh.setMatrixAt(i, dummy.matrix);
    });
    crowd.mesh.count = crowd.bodies.length;
    crowd.mesh.instanceMatrix.needsUpdate = true;
  }

  function animate() {
    const delta = Math.min(clock.getDelta(), 0.1);
    elapsedTime += delta;

    if (!gameOver) {
      timeLeft -= delta;
      if (timeLeft <= 0) {
        timeLeft = 0;
        endRound("Time's Up!");
      }

      const playerSpeed = 9;
      const keyDir = keyboardDirection();
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

      steerCrowd(player);
      rivalCrowds.forEach((r) => steerCrowd(r));

      world.step(1 / 60, delta, 3);

      renderCrowd(player);
      rivalCrowds.forEach((r) => renderCrowd(r));

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
