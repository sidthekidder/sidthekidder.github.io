// Pure, side-effect-free crowd-simulation math (formation layout, spatial
// hashing, separation forces, AABB collision, size/speed formulas) lives
// in lib/ and is unit-tested under plain Node — see tests/ and run
// `npm test` (or `node tests/<name>.test.js` directly) from this
// directory, no install needed. Everything below this point is the
// THREE.js/DOM orchestration layer: scene setup, input handling, audio,
// and the animate() loop that wires the pure lib/ functions together —
// each lib function is used here through a small same-named wrapper that
// closes over this file's live state (buildings, elapsedTime, tuning
// constants), so the wrappers are the seam between "game glue" and
// "tested logic" if you're looking for where to extend either.
import * as THREE from 'https://esm.sh/three@0.165.0';
import { GLTFLoader } from 'https://esm.sh/three@0.165.0/examples/jsm/loaders/GLTFLoader.js';
import { resolveBoxCollision } from './lib/aabb.js';
import {
  crowdRadius,
  buildingCollisionRadius,
  steerSpeedForDistance,
  shakeMagnitudeForAbsorb,
} from './lib/crowdMath.js';
import { computeFormationSlot } from './lib/formation.js';
import { buildSpatialHash as buildHash } from './lib/spatialHash.js';
import { computeSeparation as computeSeparationForce } from './lib/separation.js';
import {
  applyAdd,
  applySub,
  applyMult,
  applyDiv,
  pickupPercentText as pickupPercentTextFor,
  addGroupsAtPosition,
  removeCountFromGroups,
} from './lib/pickups.js';
// Firebase's own CDN build, not esm.sh — the modular SDK needs
// firebase-app.js and firebase-firestore.js to share one internal
// service registry, which only Firebase's own paired build guarantees;
// a generic npm-to-ESM proxy can serve them as independently bundled
// copies that don't see each other's registrations.
import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js';
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  query,
  orderBy,
  limit,
  serverTimestamp,
} from 'https://www.gstatic.com/firebasejs/10.14.1/firebase-firestore.js';

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
  // Each cluster of up to GROUP_SIZE soldiers steers/separates as one
  // simulated entity but renders as that many separate instances (small
  // fixed offsets, see GROUP_OFFSETS) — at 1, every soldier is its own
  // simulated entity.
  const GROUP_SIZE = 1;
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

  // --- Global high scores ---
  // Firestore (free Spark plan, no server of our own). This apiKey is the
  // public client identifier Firebase expects to ship in client code —
  // actual write protection is enforced server-side by Firestore
  // Security Rules (reject bad shapes/out-of-range scores, no update or
  // delete), not by keeping this value secret.
  const firebaseConfig = {
    apiKey: 'AIzaSyA8Fg0TJ6pMumXMxt6s2Nn8CLJ4l0-L8FY',
    authDomain: 'crowdrush-1f668.firebaseapp.com',
    projectId: 'crowdrush-1f668',
    storageBucket: 'crowdrush-1f668.firebasestorage.app',
    messagingSenderId: '576644056428',
    appId: '1:576644056428:web:00779b80e733150b6b5f49',
  };
  const firestore = getFirestore(initializeApp(firebaseConfig));
  const scoresCollection = collection(firestore, 'scores');
  const HIGH_SCORE_COUNT = 10;
  // Every submission is kept in Firestore (the rules are append-only, no
  // updates/deletes, to keep the security model simple), so the same
  // name can have many entries — fetch more candidates than we need and
  // keep only each name's best one when displaying.
  const HIGH_SCORE_FETCH_COUNT = 50;
  const HIGH_SCORE_NAME_KEY = 'crowdRushName';

  const scoreNameInput = document.getElementById('score-name-input');
  const submitScoreBtn = document.getElementById('submit-score-btn');
  const scoreSubmitStatus = document.getElementById('score-submit-status');
  const highScoresList = document.getElementById('high-scores-list');

  scoreNameInput.value = localStorage.getItem(HIGH_SCORE_NAME_KEY) || '';

  function dedupeByName(docsData) {
    const seen = new Set();
    const result = [];
    for (const data of docsData) {
      const key = data.name.trim().toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      result.push(data);
      if (result.length >= HIGH_SCORE_COUNT) break;
    }
    return result;
  }

  async function refreshHighScores() {
    const q = query(scoresCollection, orderBy('score', 'desc'), limit(HIGH_SCORE_FETCH_COUNT));
    const snapshot = await getDocs(q);
    const topScores = dedupeByName(snapshot.docs.map((doc) => doc.data()));
    highScoresList.innerHTML = topScores
      .map((data, i) => `<li><span><span class="rank">${i + 1}.</span>${data.name}</span><span>${data.score}</span></li>`)
      .join('');
  }
  refreshHighScores().catch((err) => console.warn('Could not load high scores:', err));

  submitScoreBtn.addEventListener('click', async () => {
    const name = scoreNameInput.value.trim().slice(0, 20) || 'Anonymous';
    localStorage.setItem(HIGH_SCORE_NAME_KEY, name);
    submitScoreBtn.disabled = true;
    scoreSubmitStatus.textContent = 'Submitting…';
    try {
      await addDoc(scoresCollection, { name, score: Math.round(player.count), ts: serverTimestamp() });
      scoreSubmitStatus.textContent = 'Submitted!';
      await refreshHighScores();
    } catch (err) {
      console.warn('Could not submit score:', err);
      scoreSubmitStatus.textContent = 'Could not submit — try again?';
      submitScoreBtn.disabled = false;
    }
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

  function playPickupGoodSound() {
    ensureAudioResumed();
    playTone(500, 900, 0.18, 'sine', 0.18);
  }

  function playPickupBadSound() {
    ensureAudioResumed();
    playTone(500, 240, 0.22, 'sawtooth', 0.15);
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
  const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
  sunLight.position.set(20, 30, 10);
  scene.add(sunLight);

  // --- Toon shading ---
  // Everything in the scene uses MeshToonMaterial (built into Three.js
  // core, no shader code or extra CDN package needed), quantized against
  // this hand-rolled 4-step gradient map so lighting reads as flat
  // cartoon bands — needs zero external assets, same as everything else
  // on this page.
  function makeToonGradientTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 4;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    ['#4d4d4d', '#8a8a8a', '#c2c2c2', '#ffffff'].forEach((color, i) => {
      ctx.fillStyle = color;
      ctx.fillRect(i, 0, 1, 1);
    });
    const texture = new THREE.CanvasTexture(canvas);
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    return texture;
  }
  const toonGradientMap = makeToonGradientTexture();

  function makeToonMaterial(options) {
    return new THREE.MeshToonMaterial({ ...options, gradientMap: toonGradientMap });
  }

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(ARENA_HALF * 2 + 20, ARENA_HALF * 2 + 20),
    makeToonMaterial({ color: 0x9aa3ad })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // --- City blocks: a different mix of buildings/parks/open plazas every
  // time the page loads (Play Again reloads the page, so every round gets
  // a fresh layout for free). Groups avoid buildings with the same cheap
  // AABB push-out used for the logical leader position below; parks and
  // open cells have no collision at all.

  const buildings = []; // { minX, maxX, minZ, maxZ } — buildings only
  const buildingColors = [0xf2d7a0, 0xa7c7e7, 0xf4a6a6, 0xb8e0c2];
  const gridPositions = [-24, -12, 0, 12, 24];

  function makeTree(x, z) {
    const trunkHeight = 0.6 + Math.random() * 0.3;
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.1, trunkHeight, 6),
      makeToonMaterial({ color: 0x8b6b4a, flatShading: true })
    );
    trunk.position.set(x, trunkHeight / 2, z);
    scene.add(trunk);

    const foliageHeight = 1.2 + Math.random() * 0.6;
    const foliage = new THREE.Mesh(
      new THREE.ConeGeometry(0.55 + Math.random() * 0.2, foliageHeight, 7),
      makeToonMaterial({ color: 0x4a8f5c, flatShading: true })
    );
    foliage.position.set(x, trunkHeight + foliageHeight / 2 - 0.05, z);
    scene.add(foliage);
  }

  function makeParkCell(bx, bz, size) {
    const patch = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      makeToonMaterial({ color: 0x8fd18f })
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
          makeToonMaterial({ color, transparent: true, opacity: 0.55, flatShading: true })
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
  // per-frame building avoidance (see updateCrowdMotion below). Thin
  // wrapper over the pure, unit-tested resolveBoxCollision (lib/aabb.js)
  // that closes over this game's buildings list and arena size, so every
  // existing call site below keeps working unchanged.
  function resolveBuildingCollision(position, radius) {
    resolveBoxCollision(position, radius, buildings, ARENA_HALF);
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

  // --- Crowd character models ---
  // Real low-poly character models — CC0, Kenney "Mini Characters" pack,
  // see assets/characters/License.txt. These are rigged/animated source
  // files, but this game has no
  // per-instance skeletal animation (everything is one InstancedMesh per
  // crowd driven by our own steering/separation math, not bones), so
  // loadCharacterAsset below extracts just the bind-pose mesh data —
  // ignoring joints/weights/animations entirely — and merges each
  // character's multiple mesh parts (body-mesh, head-mesh) into a single
  // BufferGeometry, since InstancedMesh needs exactly one.

  const CHARACTER_FILES = [
    'character-male-a',
    'character-female-a',
    'character-male-b',
    'character-female-b',
    'character-male-c',
    'character-female-c',
  ];
  // Matches the formation spacing/collision-radius tuning elsewhere,
  // which assumes roughly this height.
  const CHARACTER_TARGET_HEIGHT = 0.6;
  // These models are exported from Unity, whose forward-axis convention
  // doesn't always match glTF's -Z-forward default — if characters appear
  // to walk backward/sideways, flip this to Math.PI (or +/- Math.PI/2).
  const CHARACTER_FORWARD_OFFSET = 0;

  // Concatenates several geometries' position/normal/uv attributes (with
  // index values offset per geometry) into one BufferGeometry — the
  // "merge multiple mesh parts into one" step InstancedMesh needs. Pure
  // Three.js core BufferGeometry/BufferAttribute APIs, no addon import.
  function mergeMeshGeometries(geometries) {
    let vertexCount = 0;
    let indexCount = 0;
    geometries.forEach((g) => {
      vertexCount += g.attributes.position.count;
      indexCount += g.index ? g.index.count : g.attributes.position.count;
    });

    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(indexCount);

    let vertexOffset = 0;
    let indexOffset = 0;

    geometries.forEach((g) => {
      const vertCount = g.attributes.position.count;

      positions.set(g.attributes.position.array, vertexOffset * 3);
      if (g.attributes.normal) {
        normals.set(g.attributes.normal.array, vertexOffset * 3);
      }
      if (g.attributes.uv) {
        uvs.set(g.attributes.uv.array, vertexOffset * 2);
      }

      if (g.index) {
        const idxArray = g.index.array;
        for (let i = 0; i < idxArray.length; i++) {
          indices[indexOffset + i] = idxArray[i] + vertexOffset;
        }
        indexOffset += idxArray.length;
      } else {
        for (let i = 0; i < vertCount; i++) {
          indices[indexOffset + i] = i + vertexOffset;
        }
        indexOffset += vertCount;
      }

      vertexOffset += vertCount;
    });

    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    merged.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    merged.setIndex(new THREE.BufferAttribute(indices, 1));
    return merged;
  }

  // Loads one character's .glb, extracts+merges its static bind-pose mesh
  // geometry (see mergeMeshGeometries), then normalizes scale to
  // CHARACTER_TARGET_HEIGHT and translates it so it stands with its feet
  // on local y=0, x/z centered — matching how the old capsule geometry
  // was always authored to sit and center on its own origin.
  function loadCharacterAsset(name) {
    return new Promise((resolve, reject) => {
      const loader = new GLTFLoader();
      loader.load(
        `assets/characters/${name}.glb`,
        (gltf) => {
          gltf.scene.updateMatrixWorld(true);
          const parts = [];
          let texture = null;
          gltf.scene.traverse((node) => {
            if (node.isMesh && node.geometry) {
              const geom = node.geometry.clone();
              geom.applyMatrix4(node.matrixWorld);
              parts.push(geom);
              if (!texture && node.material && node.material.map) {
                texture = node.material.map;
              }
            }
          });

          const merged = mergeMeshGeometries(parts);

          merged.computeBoundingBox();
          const rawHeight = Math.max(
            merged.boundingBox.max.y - merged.boundingBox.min.y,
            0.0001
          );
          const scale = CHARACTER_TARGET_HEIGHT / rawHeight;
          merged.scale(scale, scale, scale);

          merged.computeBoundingBox();
          const box = merged.boundingBox;
          merged.translate(
            -(box.max.x + box.min.x) / 2,
            -box.min.y,
            -(box.max.z + box.min.z) / 2
          );

          resolve({ geometry: merged, texture });
        },
        undefined,
        (error) => reject(error)
      );
    });
  }

  // Small fixed offsets so a GROUP_SIZE cluster reads as a few separate
  // people huddled together.
  const GROUP_OFFSETS = [
    { x: 0, z: 0 },
    { x: 0.22, z: 0.1 },
    { x: -0.18, z: 0.15 },
    { x: 0.05, z: -0.2 },
  ];

  let elapsedTime = 0;
  // crowdRadius / buildingCollisionRadius imported from lib/crowdMath.js.

  // Thin wrapper over the pure, unit-tested computeFormationSlot
  // (lib/formation.js) that injects this game's live elapsedTime clock.
  // Writing into `out` instead of allocating matters here: this runs once
  // per group every frame, and per-frame allocation (here and in
  // crowdCentroid) causes GC-pause stutter.
  function computeSlotPosition(index, centerX, centerZ, facingAngle, moveBlend, out) {
    return computeFormationSlot(index, centerX, centerZ, facingAngle, moveBlend, elapsedTime, out);
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

  // A brief squash-and-stretch pulse on absorb: every instance in the
  // crowd scales up then settles back to 1 over ABSORB_PULSE_DURATION,
  // via a single decaying sine bump keyed off `elapsedTime` (same
  // snapshot-based timing as the per-instance bob/lean in renderCrowd,
  // so no extra delta plumbing is needed). Applied per-instance in
  // renderCrowd (not on the InstancedMesh itself), since the mesh's own
  // transform always stays at identity — instance positions are already
  // world-space, so scaling the mesh would scale their distance from
  // world origin too, not just their size.
  const ABSORB_PULSE_DURATION = 0.35;

  function triggerAbsorbPulse(crowd) {
    crowd.absorbPulseStart = elapsedTime;
  }

  function absorbPulseScale(crowd) {
    if (crowd.absorbPulseStart === undefined) return 1;
    const t = (elapsedTime - crowd.absorbPulseStart) / ABSORB_PULSE_DURATION;
    if (t >= 1) {
      crowd.absorbPulseStart = undefined;
      return 1;
    }
    return 1 + Math.sin(t * Math.PI) * (1 - t) * 0.4;
  }

  // Cartoon-style outline: a second InstancedMesh sharing the same
  // geometry, rendered solid black with only its back faces visible and
  // scaled up slightly larger than the real mesh (the classic "inverted
  // hull" technique — draw the model twice, inflate the second copy
  // along its own surface, and only its back faces peek out from behind
  // the first copy's front faces, reading as an outline). No
  // postprocessing pass needed, so it costs nothing beyond one extra
  // cheap (unlit, no toon shading) InstancedMesh per crowd. The character
  // models are small and detailed (thin arms/legs/head), so this needs to
  // stay small or the outline hull swallows most of the model in black.
  const OUTLINE_SCALE = 1.045;

  function makeOutlineMesh(geometry, capacity) {
    const material = new THREE.MeshBasicMaterial({ color: 0x1a1a1a, side: THREE.BackSide });
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.frustumCulled = false;
    scene.add(mesh);
    return mesh;
  }

  function makeCrowdMesh(geometry, texture, color, capacity) {
    // These meshes carry real authored vertex normals, so flatShading
    // stays off to preserve smooth cartoon bands (unlike the primitive
    // shapes elsewhere in the scene, which use flatShading: true).
    // Tint is blended toward white so the texture's own skin/hair/
    // clothing variation stays visible per crowd color.
    const tint = new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.6);
    const material = makeToonMaterial({ color: tint, map: texture || null, flatShading: false });
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
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

  // Loaded once up front so every crowd (player + all rivals) can be
  // created synchronously below — an InstancedMesh needs its geometry at
  // construction time, so we wait for every .glb to finish before
  // building any crowd.
  const characterAssets = await Promise.all(CHARACTER_FILES.map(loadCharacterAsset));
  function characterFor(index) {
    return characterAssets[index % characterAssets.length];
  }

  // --- Player ---

  const playerCharacter = characterFor(0);
  const player = {
    position: new THREE.Vector3(0, 0, 0),
    count: 15,
    facing: 0,
    moveBlend: 0,
    mesh: makeCrowdMesh(playerCharacter.geometry, playerCharacter.texture, 0x3a7bd5, PLAYER_CAP),
    outlineMesh: makeOutlineMesh(playerCharacter.geometry, PLAYER_CAP),
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
    const rivalCharacter = characterFor(i + 1);
    const mesh = makeCrowdMesh(rivalCharacter.geometry, rivalCharacter.texture, rivalColors[i], 300);
    const outlineMesh = makeOutlineMesh(rivalCharacter.geometry, 300);
    const groups = spawnGroups(count, pos.x, pos.z);
    rivalCrowds.push({
      name: rivalNames[i],
      color: rivalColors[i],
      position: pos,
      count,
      facing: 0,
      moveBlend: 0,
      mesh,
      outlineMesh,
      groups,
      wanderTarget: pos.clone(),
      wanderTimer: 0,
    });
  }

  // --- Pickups: risk/reward orbs scattered on the roads. Collectible by
  // any crowd (player or rival — a shared risk, not a player-only
  // snowball), uniform odds across all 4 effects. Count is always
  // clamped to a minimum of 1 so a bad pickup can't zero a crowd out.
  // The effect math (scaledPickupDelta, apply*, pickupPercentText) and
  // the group add/remove mutation live in lib/pickups.js, unit-tested —
  // everything here is THREE.js/DOM presentation around that.

  const PICKUP_TYPES = [
    { key: 'add', color: 0x4ade80, apply: applyAdd },
    { key: 'sub', color: 0xef4444, apply: applySub },
    { key: 'mult', color: 0xf5c518, apply: applyMult },
    { key: 'div', color: 0xa855f7, apply: applyDiv },
  ];
  const PICKUP_COUNT = 7;
  const PICKUP_RADIUS = 0.9;
  const PICKUP_RESPAWN_DELAY = 4;
  const PICKUP_LABEL_REFRESH_INTERVAL = 0.5;

  const pickupGeometry = new THREE.OctahedronGeometry(0.35, 0);
  const activePickups = []; // { type, root, gem, glow, label, x, z, spinSeed }
  const pickupRespawnTimers = [];
  let pickupLabelRefreshTimer = 0;

  function pickupPercentText(type, referenceCount) {
    return pickupPercentTextFor(type.apply, referenceCount);
  }

  // A small always-camera-facing canvas-texture sprite.
  function makePickupLabel() {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 80;
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
    sprite.scale.set(1.3, 0.65, 1);
    sprite.position.set(0, 0.75, 0);
    return { sprite, canvas, texture };
  }

  function drawPickupLabel(label, text) {
    const ctx = label.canvas.getContext('2d');
    ctx.clearRect(0, 0, label.canvas.width, label.canvas.height);
    ctx.font = 'bold 44px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 8;
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.strokeText(text, label.canvas.width / 2, label.canvas.height / 2);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, label.canvas.width / 2, label.canvas.height / 2);
    label.texture.needsUpdate = true;
  }

  // Labels show the effect relative to the player's current count — the
  // only count a human reads a label to decide about; rivals ignore them.
  function refreshPickupLabels() {
    activePickups.forEach((p) => {
      drawPickupLabel(p.label, pickupPercentText(p.type, player.count));
    });
  }

  // Soft halo behind the gem — a radial-gradient canvas sprite with
  // additive blending, the standard cheap "glow" trick that needs no
  // postprocessing bloom pass.
  function makeGlowSprite(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const c = new THREE.Color(color);
    const rgb = `${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(c.b * 255)}`;
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, `rgba(${rgb}, 0.9)`);
    gradient.addColorStop(1, `rgba(${rgb}, 0)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    sprite.scale.set(1.6, 1.6, 1);
    return sprite;
  }

  function spawnPickup() {
    const type = PICKUP_TYPES[Math.floor(Math.random() * PICKUP_TYPES.length)];
    const pos = randomRoadPosition(0.5);

    const root = new THREE.Group();
    root.position.set(pos.x, 0.6, pos.z);

    const glow = makeGlowSprite(type.color);
    root.add(glow);

    const gem = new THREE.Mesh(pickupGeometry, new THREE.MeshBasicMaterial({ color: type.color }));
    root.add(gem);

    const label = makePickupLabel();
    drawPickupLabel(label, pickupPercentText(type, player.count));
    root.add(label.sprite);

    scene.add(root);
    activePickups.push({ type, root, gem, glow, label, x: pos.x, z: pos.z, spinSeed: Math.random() * Math.PI * 2 });
  }

  for (let i = 0; i < PICKUP_COUNT; i++) spawnPickup();

  function removePickup(pickup) {
    scene.remove(pickup.root);
    pickup.gem.material.dispose();
    pickup.glow.material.map.dispose();
    pickup.glow.material.dispose();
    pickup.label.texture.dispose();
    pickup.label.sprite.material.dispose();
    activePickups.splice(activePickups.indexOf(pickup), 1);
    pickupRespawnTimers.push(PICKUP_RESPAWN_DELAY);
  }

  function updatePickups(delta) {
    activePickups.forEach((p) => {
      p.gem.rotation.y += delta * 1.5;
      p.root.position.y = 0.6 + Math.sin(elapsedTime * 2 + p.spinSeed) * 0.12;
      const pulse = 1 + Math.sin(elapsedTime * 3 + p.spinSeed) * 0.15;
      p.gem.scale.setScalar(pulse);
      p.glow.scale.set(1.6 * pulse, 1.6 * pulse, 1);
    });

    pickupLabelRefreshTimer -= delta;
    if (pickupLabelRefreshTimer <= 0) {
      pickupLabelRefreshTimer = PICKUP_LABEL_REFRESH_INTERVAL;
      refreshPickupLabels();
    }

    for (let i = pickupRespawnTimers.length - 1; i >= 0; i--) {
      pickupRespawnTimers[i] -= delta;
      if (pickupRespawnTimers[i] <= 0) {
        pickupRespawnTimers.splice(i, 1);
        spawnPickup();
      }
    }
  }

  const pickupCollisionCentroidScratch = { x: 0, z: 0 };

  function resolvePickupsForCrowd(crowd) {
    const centroid = crowdCentroid(crowd, pickupCollisionCentroidScratch);
    for (let i = activePickups.length - 1; i >= 0; i--) {
      const p = activePickups[i];
      const dist = Math.hypot(centroid.x - p.x, centroid.z - p.z);
      if (dist < crowdRadius(crowd.count) + PICKUP_RADIUS) {
        const oldCount = crowd.count;
        const newCount = Math.max(1, p.type.apply(oldCount));
        crowd.count = newCount;
        const delta = newCount - oldCount;
        if (delta > 0) {
          addGroupsAtPosition(crowd.groups, delta, p.x, p.z, GROUP_SIZE);
        } else if (delta < 0) {
          removeCountFromGroups(crowd.groups, -delta);
        }
        spawnBurst(p.x, p.z, p.type.color);
        if (p.type.key === 'add' || p.type.key === 'mult') {
          playPickupGoodSound();
        } else {
          playPickupBadSound();
        }
        removePickup(p);
      }
    }
  }

  function checkPickupCollisions() {
    resolvePickupsForCrowd(player);
    rivalCrowds.forEach((r) => resolvePickupsForCrowd(r));
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

  // Touch devices get a further-back base view (same screen space has to
  // show the same arena on a much smaller display), independent of the
  // crowd-size zoom below.
  const isTouchDevice = window.matchMedia('(pointer: coarse)').matches;
  const MOBILE_ZOOM_OUT = isTouchDevice ? 1.35 : 1;

  // A brief decaying random jitter on top of the normal camera position,
  // for impact on big absorptions. `magnitude` scales with how big the
  // steal was (see lib/crowdMath.js's shakeMagnitudeForAbsorb), and
  // decays linearly to 0 over CAMERA_SHAKE_DURATION.
  const CAMERA_SHAKE_DURATION = 0.3;
  let cameraShakeTimer = 0;
  let cameraShakeMagnitude = 0;

  function triggerCameraShake(magnitude) {
    cameraShakeTimer = CAMERA_SHAKE_DURATION;
    cameraShakeMagnitude = magnitude;
  }

  // `target` defaults to the player — but during the defeat sequence
  // (see pendingDefeat in animate()) the camera follows whichever rival
  // just absorbed the player instead, since player.groups is empty by
  // then and would otherwise centroid to nothing.
  function updateCamera(target, delta) {
    const crowd = target || player;
    const baseBehind = 14 * MOBILE_ZOOM_OUT;
    const baseHeight = 16 * MOBILE_ZOOM_OUT;
    const centroid = crowdCentroid(crowd, cameraCentroidScratch);

    const targetZoom = Math.min(
      2,
      1 + Math.max(0, crowdRadius(crowd.count) - crowdRadius(15)) * 0.12
    );
    const zoomEase = Math.min(1, (delta === undefined ? 1 : delta) * 1.2);
    cameraZoom += (targetZoom - cameraZoom) * zoomEase;

    camera.position.set(centroid.x, baseHeight * cameraZoom, centroid.z + baseBehind * cameraZoom);

    if (cameraShakeTimer > 0) {
      cameraShakeTimer -= delta === undefined ? 0 : delta;
      const strength = cameraShakeMagnitude * Math.max(0, cameraShakeTimer / CAMERA_SHAKE_DURATION);
      camera.position.x += (Math.random() - 0.5) * strength;
      camera.position.y += (Math.random() - 0.5) * strength * 0.5;
      camera.position.z += (Math.random() - 0.5) * strength;
    }

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

  function isTextInput(target) {
    return target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
  }

  window.addEventListener('keydown', (e) => {
    if (isTextInput(e.target)) return;
    if (movementKeyCodes.has(e.code)) e.preventDefault();
    setKeyState(e.code, true);
  });
  window.addEventListener('keyup', (e) => {
    if (isTextInput(e.target)) return;
    setKeyState(e.code, false);
  });

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

  // --- Touch input: a floating virtual joystick, usable from anywhere on
  // screen — a touch starting on empty game area spawns it centered on
  // that point (CSS positions/animates it via left/top + .is-active).
  // Touches on real UI (back link, end-of-round screen) are excluded so
  // those stay tappable normally, and only touch pointers trigger it —
  // mouse/pen pointers fall through to native behavior.

  const joystickEl = document.getElementById('joystick');
  const joystickKnobEl = document.getElementById('joystick-knob');
  const JOYSTICK_MAX_RADIUS = 45;
  const JOYSTICK_DEADZONE = 8;

  let joystickPointerId = null;
  let joystickOriginX = 0;
  let joystickOriginY = 0;
  let joystickDirX = 0;
  let joystickDirZ = 0;

  function isJoystickExcluded(target) {
    return !!target.closest('a, button, .end-screen');
  }

  function updateJoystickFromEvent(clientX, clientY) {
    let dx = clientX - joystickOriginX;
    let dy = clientY - joystickOriginY;
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
    joystickEl.classList.remove('is-active');
    joystickKnobEl.style.transform = 'translate(-50%, -50%)';
  }

  window.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'touch' || joystickPointerId !== null || isJoystickExcluded(e.target)) {
      return;
    }
    joystickPointerId = e.pointerId;
    joystickOriginX = e.clientX;
    joystickOriginY = e.clientY;
    joystickEl.style.left = `${joystickOriginX}px`;
    joystickEl.style.top = `${joystickOriginY}px`;
    joystickEl.classList.add('is-active');
    e.target.setPointerCapture(e.pointerId);
  });
  window.addEventListener('pointermove', (e) => {
    if (e.pointerId === joystickPointerId) {
      updateJoystickFromEvent(e.clientX, e.clientY);
    }
  });
  window.addEventListener('pointerup', (e) => {
    if (e.pointerId === joystickPointerId) resetJoystick();
  });
  window.addEventListener('pointercancel', (e) => {
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

  // Shared pause before either end-of-round screen actually shows, so the
  // round never just freezes instantly — see pendingDefeat/pendingVictory
  // below for what plays during it in each case.
  const ROUND_END_DELAY = 1;
  // Losing: the player's groups transfer into the winning rival (visually
  // "converting" into its crowd/color) and the camera follows that rival
  // for the delay.
  let pendingDefeat = null; // { rival, timer } while the sequence plays out
  // Winning by absorbing the last rival: the player keeps playing/
  // rendering normally (already grew on absorption) for the same delay,
  // camera staying on the player, before the end screen shows.
  let pendingVictory = null; // { timer } while the sequence plays out

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
  // array (no destroy/recreate — they're plain state), so the absorbed
  // soldiers keep their exact position/velocity and visibly run to join
  // the winner's new formation on later frames.
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
            triggerAbsorbPulse(winner);
            winner.groups = winner.groups.concat(loser.groups);
            winner.count += loser.count;
            scene.remove(loser.mesh);
            scene.remove(loser.outlineMesh);
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
          triggerAbsorbPulse(player);
          if (r.count >= 6) triggerCameraShake(shakeMagnitudeForAbsorb(r.count));
          player.groups = player.groups.concat(r.groups);
          player.count += r.count;
          scene.remove(r.mesh);
          scene.remove(r.outlineMesh);
          rivalCrowds.splice(i, 1);
        } else {
          spawnBurst(playerCentroid.x, playerCentroid.z, 0xff4d4d);
          triggerAbsorbPulse(r);
          if (player.count >= 6) triggerCameraShake(shakeMagnitudeForAbsorb(player.count));
          // Player's groups visually convert into the winning rival's
          // crowd instead of just vanishing — r keeps existing (not
          // removed/spliced) so it can keep rendering them merging in
          // during the delay window below. player.count is left
          // untouched: endRound's "Final size" reads it, and zeroing it
          // here would show 0 instead of the size you actually reached.
          const absorbedGroups = player.groups;
          r.groups = r.groups.concat(absorbedGroups);
          r.count += player.count;
          player.groups = [];
          pendingDefeat = { rival: r, timer: ROUND_END_DELAY };
          return;
        }
      }
    }

    if (rivalCrowds.length === 0) {
      pendingVictory = { timer: ROUND_END_DELAY };
    }
  }

  // --- Spatial hash + boids-style separation ---
  // Gives jostle/knockback: groups (regardless of which crowd owns them)
  // push apart from close neighbors within SEPARATION_RADIUS, found via
  // the spatial hash's 3x3 cell neighborhood query (lib/spatialHash.js,
  // lib/separation.js). Rebuilt fresh every frame (cheap — O(number of
  // groups)) rather than incrementally maintained, since groups move
  // every frame anyway. CELL_SIZE is a little larger than
  // SEPARATION_RADIUS so a 3x3 cell neighborhood always covers it.

  const CELL_SIZE = 1.2;
  const SEPARATION_RADIUS = 0.9;
  const SEPARATION_STRENGTH = 5;
  const separationScratch = { x: 0, z: 0 };

  function buildSpatialHash(allGroups) {
    return buildHash(allGroups, CELL_SIZE);
  }

  function computeSeparation(group, hash, out) {
    return computeSeparationForce(group, hash, CELL_SIZE, SEPARATION_RADIUS, SEPARATION_STRENGTH, out);
  }

  const slotScratch = { x: 0, z: 0 };
  // steerSpeedForDistance imported from lib/crowdMath.js.

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
      // mechanism, so a group never visibly pops from one position to
      // another, even if stuck against an awkward building corner.
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
  // cosmetic bob/lean, so a cluster still reads as a few separate people
  // — and a matching, slightly larger set into its outline companion
  // mesh (see makeOutlineMesh) at the same position/rotation.
  function renderCrowd(crowd) {
    const pulse = absorbPulseScale(crowd);
    let renderIndex = 0;
    crowd.groups.forEach((group) => {
      for (let k = 0; k < group.size; k++) {
        const offset = GROUP_OFFSETS[k];
        const seed = renderIndex * 12.9898;
        const bob = Math.abs(Math.sin(elapsedTime * 7 + seed)) * 0.14;
        const lean = Math.sin(elapsedTime * 5 + seed) * 0.12;
        dummy.position.set(
          group.position.x + offset.x,
          bob,
          group.position.z + offset.z
        );
        dummy.rotation.y = crowd.facing + lean + CHARACTER_FORWARD_OFFSET;

        dummy.scale.setScalar(pulse);
        dummy.updateMatrix();
        crowd.mesh.setMatrixAt(renderIndex, dummy.matrix);

        dummy.scale.setScalar(OUTLINE_SCALE * pulse);
        dummy.updateMatrix();
        crowd.outlineMesh.setMatrixAt(renderIndex, dummy.matrix);

        renderIndex++;
      }
    });
    crowd.mesh.count = renderIndex;
    crowd.mesh.instanceMatrix.needsUpdate = true;
    crowd.outlineMesh.count = renderIndex;
    crowd.outlineMesh.instanceMatrix.needsUpdate = true;
  }

  // Defeat/victory both pause normal play for a beat to show the final
  // absorb settle in (see pendingDefeat/pendingVictory below), running at
  // this fraction of normal speed for a deliberate dramatic effect — the
  // beat's real-world wall-clock length stretches accordingly.
  const END_SEQUENCE_SLOWMO = 0.35;

  function animate() {
    const slowMo = pendingDefeat || pendingVictory ? END_SEQUENCE_SLOWMO : 1;
    const delta = Math.min(clock.getDelta(), 0.1) * slowMo;
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
    } else if (pendingVictory) {
      // Same beat as pendingDefeat, mirrored: the player already grew
      // from absorbing the last rival, so there's nothing to transfer —
      // just keep letting that final absorption settle into formation,
      // camera staying on the player, before the end screen shows.
      pendingVictory.timer -= delta;

      const hash = buildSpatialHash(player.groups);
      updateCrowdMotion(player, hash, delta);
      renderCrowd(player);

      updateLeaderboard();
      updateCamera(player, delta);

      if (pendingVictory.timer <= 0) {
        pendingVictory = null;
        endRound('All Rivals Defeated!');
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
      checkPickupCollisions();

      checkCollisions();

      // checkCollisions may have just set pendingDefeat (and cleared
      // player.groups) or pendingVictory as a side effect — skip the rest
      // of this frame's normal-path rendering/camera in that case so
      // updateCamera doesn't centroid an empty player crowd to world
      // origin for one frame before the pending* branch takes over next.
      if (!pendingDefeat && !pendingVictory) {
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
    updatePickups(delta);

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  updateLeaderboard();
  animate();
}
