#!/usr/bin/env node
/**
 * Headless WASM node — runs the Distrain training engine in Node.js.
 *
 * This script participates in distributed training the same way a browser node
 * would, but runs headless (no browser UI). It:
 *   1. Registers with the coordinator
 *   2. Gets the latest checkpoint info
 *   3. Initializes the WASM training engine
 *   4. Trains for N steps, then computes a delta
 *   5. Uploads the raw JSON delta to MinIO/R2 (no zstd — WASM can't link it)
 *   6. Pushes delta metadata to the coordinator
 *   7. Loops
 *
 * Usage:
 *   node scripts/wasm_node_headless.mjs [--coordinator http://localhost:8000] [--steps 10]
 *
 * Prerequisites:
 *   - wasm-pack build crates/wasm --target nodejs --out-dir ../../web/pkg-node
 *   - npm install @aws-sdk/client-s3  (or use the project's package.json)
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');

// --- Parse args ---
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(name);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : defaultVal;
}

const COORDINATOR_URL = getArg('--coordinator', 'http://localhost:8000');
const INNER_STEPS = parseInt(getArg('--steps', '10'), 10);
const MAX_ROUNDS = parseInt(getArg('--rounds', '100'), 10);
const MINIO_ENDPOINT = getArg('--s3-endpoint', 'http://localhost:9000');
const MINIO_BUCKET = getArg('--bucket', 'distrain-training');
const MINIO_ACCESS_KEY = getArg('--s3-access-key', 'minioadmin');
const MINIO_SECRET_KEY = getArg('--s3-secret-key', 'minioadmin');
const MODEL_PRESET = getArg('--preset', 'Tiny');

// --- Load WASM module ---
let wasm;
try {
  // Try Node.js target build first
  wasm = await import(resolve(PROJECT_ROOT, 'web/pkg-node/distrain_wasm.js'));
  console.log('[wasm] Loaded WASM module (Node.js target)');
} catch (e) {
  console.error('[wasm] Failed to load WASM module.');
  console.error('       Build with: wasm-pack build crates/wasm --target nodejs --out-dir ../../web/pkg-node');
  console.error('       Error:', e.message);
  process.exit(1);
}

// --- S3 client ---
let s3Client;
try {
  const { S3Client, PutObjectCommand } = await import('@aws-sdk/client-s3');
  s3Client = new S3Client({
    endpoint: MINIO_ENDPOINT,
    region: 'us-east-1',
    credentials: {
      accessKeyId: MINIO_ACCESS_KEY,
      secretAccessKey: MINIO_SECRET_KEY,
    },
    forcePathStyle: true,
  });
  var PutCommand = PutObjectCommand;
  console.log('[s3] S3 client initialized');
} catch (e) {
  console.error('[s3] @aws-sdk/client-s3 not found. Install with: npm install @aws-sdk/client-s3');
  process.exit(1);
}

// --- Helper: HTTP fetch with JSON ---
async function fetchJson(url, opts = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

// --- Register with coordinator ---
async function registerNode() {
  try {
    const data = await fetchJson(`${COORDINATOR_URL}/nodes/register`, {
      method: 'POST',
      body: JSON.stringify({
        node_type: 'wasm',
        capabilities: { backend: 'wasm-ndarray', platform: 'node.js' },
      }),
    });
    console.log(`[coord] Registered as node: ${data.node_id || 'ok'}`);
    return data;
  } catch (e) {
    console.warn(`[coord] Registration failed (may be optional): ${e.message}`);
    return { node_id: `wasm-${Date.now()}` };
  }
}

// --- Get latest checkpoint info ---
async function getCheckpointInfo() {
  try {
    return await fetchJson(`${COORDINATOR_URL}/checkpoint/latest`);
  } catch (e) {
    console.warn(`[coord] Failed to get checkpoint info: ${e.message}`);
    return { version: 0 };
  }
}

// --- Upload delta to S3/MinIO ---
async function uploadDelta(nodeId, seqNum, version, deltaJson) {
  const key = `deltas/v${version}/${nodeId}_${seqNum}.delta.zst`;
  // Note: we upload raw JSON even though the key says .zst
  // The aggregator handles both formats
  const body = new TextEncoder().encode(deltaJson);

  await s3Client.send(new PutCommand({
    Bucket: MINIO_BUCKET,
    Key: key,
    Body: body,
    ContentType: 'application/json',
  }));

  console.log(`[s3] Uploaded delta: ${key} (${body.length} bytes)`);
  return key;
}

// --- Push delta metadata to coordinator ---
async function pushDelta(nodeId, seqNum, version, s3Key, innerSteps, weight) {
  try {
    return await fetchJson(`${COORDINATOR_URL}/delta`, {
      method: 'POST',
      body: JSON.stringify({
        node_id: nodeId,
        seq_num: seqNum,
        checkpoint_version: version,
        s3_key: s3Key,
        inner_steps: innerSteps,
        weight: weight,
        loss: 0,
      }),
    });
  } catch (e) {
    console.warn(`[coord] Failed to push delta: ${e.message}`);
    return null;
  }
}

// --- Main training loop ---
async function main() {
  console.log(`\n=== Distrain WASM Headless Node ===`);
  console.log(`Coordinator: ${COORDINATOR_URL}`);
  console.log(`Inner steps: ${INNER_STEPS}`);
  console.log(`Max rounds: ${MAX_ROUNDS}`);
  console.log(`Model preset: ${MODEL_PRESET}\n`);

  // Register
  const reg = await registerNode();
  const nodeId = reg.node_id || `wasm-${Date.now()}`;

  // Initialize model
  console.log(`[wasm] Initializing ${MODEL_PRESET} model...`);
  const initResult = JSON.parse(wasm.wasm_init(MODEL_PRESET));
  if (initResult.error) {
    console.error(`[wasm] Init failed: ${initResult.error}`);
    process.exit(1);
  }
  console.log(`[wasm] Model initialized`);

  let seqNum = 0;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    // Get checkpoint info
    const ckpt = await getCheckpointInfo();
    const version = ckpt.version || 0;

    // Snapshot params before training
    const snapResult = JSON.parse(wasm.wasm_snapshot_params());
    if (snapResult.error) {
      console.error(`[wasm] Snapshot failed: ${snapResult.error}`);
      break;
    }
    console.log(`[round ${round}] Snapshotted ${snapResult.num_params} params, training ${INNER_STEPS} steps...`);

    // Train for inner_steps
    let lastLoss = 0;
    for (let s = 0; s < INNER_STEPS; s++) {
      const result = JSON.parse(wasm.wasm_train_step(3e-4));
      if (result.error) {
        console.error(`[wasm] Training error: ${result.error}`);
        break;
      }
      lastLoss = result.loss;
    }
    console.log(`[round ${round}] Training done, loss=${lastLoss.toFixed(4)}`);

    // Compute delta
    const deltaJson = wasm.wasm_compute_delta_json();
    const parsed = JSON.parse(deltaJson);
    if (parsed.error) {
      console.error(`[wasm] Delta computation failed: ${parsed.error}`);
      continue;
    }

    // Upload delta
    const s3Key = await uploadDelta(nodeId, seqNum, version, deltaJson);

    // Push metadata to coordinator
    const weight = INNER_STEPS / 50.0;
    await pushDelta(nodeId, seqNum, version, s3Key, INNER_STEPS, weight);

    seqNum++;
    console.log(`[round ${round}] Delta pushed (seq=${seqNum}, version=${version}, weight=${weight.toFixed(2)})\n`);
  }

  wasm.wasm_shutdown();
  console.log('[wasm] Shutdown complete');
}

main().catch(e => {
  console.error('Fatal error:', e);
  process.exit(1);
});
