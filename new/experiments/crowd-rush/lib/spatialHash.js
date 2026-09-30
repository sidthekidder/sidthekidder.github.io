// A spatial hash replaces a full physics engine's broadphase for "which
// entities are near this point" queries — bucket entities into a grid,
// then only check the 3x3 neighborhood of cells around a query point
// instead of every entity. `cellSize` should be at least as large as the
// biggest interaction radius a caller will query with, so a 3x3
// neighborhood always covers it. Pure data-structure logic — entities
// only need a `{ position: { x, z } }` shape, nothing THREE.js-specific.

export function cellKey(cx, cz) {
  return cx + ':' + cz;
}

/** Buckets `entities` (each needing `.position.{x,z}`) into a fresh hash. */
export function buildSpatialHash(entities, cellSize) {
  const hash = new Map();
  for (const entity of entities) {
    const cx = Math.floor(entity.position.x / cellSize);
    const cz = Math.floor(entity.position.z / cellSize);
    const key = cellKey(cx, cz);
    let bucket = hash.get(key);
    if (!bucket) {
      bucket = [];
      hash.set(key, bucket);
    }
    bucket.push(entity);
  }
  return hash;
}

/** Calls `callback(entity)` for every entity in the 3x3 cell neighborhood around (x, z). */
export function forEachNearby(hash, x, z, cellSize, callback) {
  const cx = Math.floor(x / cellSize);
  const cz = Math.floor(z / cellSize);
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      const bucket = hash.get(cellKey(cx + dx, cz + dz));
      if (bucket) {
        for (const other of bucket) callback(other);
      }
    }
  }
}
