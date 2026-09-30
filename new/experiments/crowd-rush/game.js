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
  // Soldiers are batched into physics-body "clusters" rather than one
  // rigid body each — broadphase collision cost scales roughly with the
  // square of body count, so grouping 4 soldiers per body cuts body count
  // (and pair checks) by ~16x. Each cluster still renders GROUP_SIZE
  // separate capsules (small fixed offsets around the body), so it reads
  // as individuals, just no longer as individually-simulated physics.
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
  // Groups of soldiers (not the crowd centroid, and not one body per
  // soldier) are real rigid bodies — so clusters physically jostle each
  // other, collide with buildings, and knock into rival crowds on
  // contact, while staying well under the body count a fully
  // one-per-soldier simulation would need.

  const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.82, 0) });
  // Default broadphase (NaiveBroadphase) checks every body pair
  // regardless of distance. SAPBroadphase sorts bodies along an axis and
  // skips pairs that can't possibly be touching — meaningfully cheaper
  // once there's more than a handful of bodies spread across the arena.
  world.broadphase = new CANNON.SAPBroadphase(world);

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
  // Aligns each capsule's visual base with where its group's physics
  // sphere actually rests on the ground (sphere settles at y = its own
  // radius).
  const RENDER_Y_OFFSET = PERSON_HALF_HEIGHT - GROUP_RADIUS;

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
  // the facing direction (moveBlend 1). This is a target for physics
  // *steering*, not a direct position write — the solver has the final
  // say once collisions with neighbors/buildings are factored in.
  //
  // Writes into `out` instead of returning a fresh object — this runs
  // once per group every single frame (steerCrowd's hot path), and a
  // researched pass at the "jumping" frame rate found that repeated
  // per-frame allocation here (and in crowdCentroid) was producing
  // periodic GC pauses, which is the textbook signature of stutter/
  // jumping rather than uniformly-low FPS.
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
  // its groups' physics bodies actually are, not the abstract WASD-driven
  // intent point (crowd.position). Used for anything that should reflect
  // what's actually on screen: collision/contact checks and the camera.
  //
  // Pass `out` to write into an existing object instead of allocating —
  // matters most in checkRivalVsRivalCollisions, which used to call this
  // fresh for the same rival on every pair comparison in its O(n^2) loop.
  function crowdCentroid(crowd, out) {
    let sumX = 0;
    let sumZ = 0;
    let totalSize = 0;
    crowd.groups.forEach((group) => {
      sumX += group.body.position.x * group.size;
      sumZ += group.body.position.z * group.size;
      totalSize += group.size;
    });
    const n = totalSize || 1;
    const target = out || { x: 0, z: 0 };
    target.x = sumX / n;
    target.z = sumZ / n;
    return target;
  }

  function makeGroupBody(x, z) {
    const body = new CANNON.Body({
      mass: 1,
      shape: new CANNON.Sphere(GROUP_RADIUS),
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

  // Spawns GROUP_SIZE-soldier clusters already near their formation slot
  // (rather than all stacked at one point), so the physics solver doesn't
  // have to violently separate a pile of exactly-overlapping bodies on
  // the first step. Each group's target slot is computed at its first
  // individual index — close enough to where the rest of that cluster
  // would sit, since a few consecutive formation indices are always
  // spatially near each other.
  function spawnGroups(count, x, z) {
    const groups = [];
    let placed = 0;
    while (placed < count) {
      const size = Math.min(GROUP_SIZE, count - placed);
      const slot = computeSlotPosition(placed, x, z, 0, 0, { x: 0, z: 0 });
      groups.push({ body: makeGroupBody(slot.x, slot.z), size });
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
    // Capacity is generous headroom, not the expected size: rivals can now
    // absorb each other, so a single rival could in the worst case end up
    // holding close to the whole 10-rival pool.
    const mesh = makeCrowdMesh(rivalColors[i], 300);
    const groups = spawnGroups(count, pos.x, pos.z);
    rivalCrowds.push({
      name: rivalNames[i],
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
  // WASD-driven position — the logical point moves at a flat, undamped
  // speed with no physical resistance, so it would otherwise steadily
  // pull ahead of the real (steered, damped, collision-slowed) crowd and
  // leave the camera looking at empty space in front of the group. Using
  // the centroid of every real body (rather than just the leader) also
  // means one stuck straggler can't drag the camera off if the rest of
  // the crowd has moved on.
  const cameraCentroidScratch = { x: 0, z: 0 };

  function updateCamera() {
    const behind = 14;
    const height = 16;
    const centroid = crowdCentroid(player, cameraCentroidScratch);
    camera.position.set(centroid.x, height, centroid.z + behind);
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
      // Each rival's centroid is the same for every pair it's checked
      // against within this pass — compute it once per rival here rather
      // than recomputing it from scratch on every pair comparison below
      // (was up to ~45 redundant recomputations per pass at 10 rivals).
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
          player.groups = player.groups.concat(r.groups);
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
  const slotScratch = { x: 0, z: 0 };

  // Sets each group's steering intent (velocity toward the formation slot
  // of its first individual index) before the physics step — the solver
  // has the final say on actual motion once collisions with neighboring
  // groups/buildings are resolved.
  //
  // LEASH_DISTANCE is a hard cap: the largest formation ring in a
  // realistically-sized crowd sits around ~4.4 units out, so anything
  // past 6 isn't a group running to catch up, it's stuck on a building or
  // lost after an absorb — snap it straight back rather than let it drift
  // indefinitely and spread the crowd out further than the collision
  // radius actually represents.
  const LEASH_DISTANCE = 6;

  function steerCrowd(crowd) {
    const steerSpeed = 6;
    let individualIndex = 0;
    crowd.groups.forEach((group) => {
      const body = group.body;
      const slot = computeSlotPosition(
        individualIndex,
        crowd.position.x,
        crowd.position.z,
        crowd.facing,
        crowd.moveBlend,
        slotScratch
      );
      const dx = slot.x - body.position.x;
      const dz = slot.z - body.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist > LEASH_DISTANCE) {
        body.position.x = slot.x + (Math.random() - 0.5) * 0.3;
        body.position.z = slot.z + (Math.random() - 0.5) * 0.3;
        body.velocity.x = 0;
        body.velocity.z = 0;
      } else if (dist > 0.02) {
        const speed = Math.min(steerSpeed, dist * 8);
        body.velocity.x = (dx / dist) * speed;
        body.velocity.z = (dz / dist) * speed;
      } else {
        body.velocity.x *= 0.5;
        body.velocity.z *= 0.5;
      }
      individualIndex += group.size;
    });
  }

  // Reads back each group's resolved physics position (after the world
  // step) and writes GROUP_SIZE instances into the crowd's InstancedMesh
  // — small fixed offsets around the body plus a per-instance cosmetic
  // bob/lean, so a cluster still reads as a few separate people.
  function renderCrowd(crowd) {
    let renderIndex = 0;
    crowd.groups.forEach((group) => {
      const body = group.body;
      for (let k = 0; k < group.size; k++) {
        const offset = GROUP_OFFSETS[k];
        const seed = renderIndex * 12.9898;
        const bob = Math.abs(Math.sin(elapsedTime * 7 + seed)) * 0.14;
        const lean = Math.sin(elapsedTime * 5 + seed) * 0.12;
        dummy.position.set(
          body.position.x + offset.x,
          body.position.y + RENDER_Y_OFFSET + bob,
          body.position.z + offset.z
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

    if (!gameOver) {
      timeLeft -= delta;
      if (timeLeft <= 0) {
        timeLeft = 0;
        endRound("Time's Up!");
      }

      // Kept close to (but slightly above) steerSpeed below, so the
      // logical target leads the real crowd just enough to feel
      // responsive without the gap growing large enough to be visible.
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
