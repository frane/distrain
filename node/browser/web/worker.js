/**
 * Distrain Training Web Worker
 *
 * Runs the WASM training engine off the main thread so the UI stays responsive.
 * Communicates with the main thread via postMessage.
 *
 * Messages TO worker:
 *   { type: 'init', preset }
 *   { type: 'loadCheckpoint', data: ArrayBuffer }
 *   { type: 'loadShards', shards: ArrayBuffer[] }
 *   { type: 'startTraining', config: { batchSize, seqLen, stepsPerRound, learningRate } }
 *   { type: 'stop' }
 *   { type: 'getStatus' }
 *
 * Messages FROM worker:
 *   { type: 'log', msg }
 *   { type: 'status', step, loss, tokens, round }
 *   { type: 'delta', data: string, loss, tokens, secs }
 *   { type: 'snapshotDone' }
 *   { type: 'ready' }
 *   { type: 'error', msg }
 *   { type: 'roundComplete', round }
 *   { type: 'stopped' }
 */

let wasm = null;
let dataShards = [];
let shardOffset = 0;
let currentShardIdx = 0;
let training = false;
let step = 0;
let lastLoss = 0;
let totalTokens = 0;

/** Cosine LR schedule with linear warmup — port of core/model/src/training.rs::cosine_lr */
function cosineLr(step, warmupSteps, totalSteps, maxLr, minLr) {
  if (step < warmupSteps) {
    return maxLr * (step / warmupSteps);
  }
  const progress = (step - warmupSteps) / Math.max(totalSteps - warmupSteps, 1);
  return minLr + 0.5 * (maxLr - minLr) * (1.0 + Math.cos(Math.PI * progress));
}

function getNextBatch(batchSize, seqLen) {
  if (dataShards.length === 0) return null;
  const needed = batchSize * seqLen;
  const batch = new Uint16Array(needed);
  let filled = 0;

  while (filled < needed) {
    const shard = dataShards[currentShardIdx];
    const available = shard.length - shardOffset;
    const take = Math.min(available, needed - filled);
    batch.set(shard.subarray(shardOffset, shardOffset + take), filled);
    filled += take;
    shardOffset += take;

    if (shardOffset >= shard.length) {
      currentShardIdx = (currentShardIdx + 1) % dataShards.length;
      shardOffset = 0;
    }
  }
  return batch;
}

self.onmessage = async function(e) {
  const msg = e.data;

  switch (msg.type) {
    case 'init': {
      try {
        const wasmModule = await import('./pkg/distrain_wasm.js');
        await wasmModule.default();
        wasm = wasmModule;

        // Prefer WebGPU (GPU), fallback to CPU (ndarray)
        const hasGpu = typeof navigator !== 'undefined' && !!navigator.gpu;
        const useGpu = msg.useGpu !== undefined ? msg.useGpu : hasGpu;

        let result;
        if (useGpu) {
          try {
            self.postMessage({ type: 'log', msg: 'Backend: WebGPU (GPU)' });
            result = JSON.parse(await wasm.wasm_init(msg.preset, true));
          } catch (gpuErr) {
            self.postMessage({ type: 'log', msg: `WebGPU failed: ${gpuErr.message}, using CPU` });
            result = JSON.parse(await wasm.wasm_init(msg.preset, false));
          }
        } else {
          self.postMessage({ type: 'log', msg: 'Backend: CPU (ndarray)' });
          result = JSON.parse(await wasm.wasm_init(msg.preset, false));
        }

        if (result.error) {
          self.postMessage({ type: 'error', msg: result.error });
        } else {
          self.postMessage({ type: 'ready', preset: msg.preset, backend: result.backend, maxSeqLen: result.max_seq_len });
        }
      } catch (err) {
        self.postMessage({ type: 'error', msg: `WASM init failed: ${err.message}` });
      }
      break;
    }

    case 'loadCheckpoint': {
      if (!wasm) { self.postMessage({ type: 'error', msg: 'Not initialized' }); break; }
      const result = JSON.parse(wasm.wasm_load_checkpoint(new Uint8Array(msg.data)));
      if (result.error) {
        self.postMessage({ type: 'error', msg: result.error });
      } else {
        self.postMessage({ type: 'log', msg: `Checkpoint loaded (${result.num_params.toLocaleString()} params)` });
      }
      break;
    }

    case 'loadShards': {
      dataShards = msg.shards.map(buf => new Uint16Array(buf));
      shardOffset = 0;
      currentShardIdx = 0;
      const totalTokensLoaded = dataShards.reduce((s, d) => s + d.length, 0);
      self.postMessage({ type: 'log', msg: `Worker loaded ${dataShards.length} shards (${(totalTokensLoaded / 1e6).toFixed(1)}M tokens)` });
      break;
    }

    case 'startTraining': {
      if (!wasm || training) break;
      training = true;
      const { batchSize, seqLen, stepsPerRound, lrMax, lrMin, warmupFraction } = msg.config;
      const warmupSteps = Math.max(Math.floor(stepsPerRound * warmupFraction), 1);

      for (let round = 0; round < 10000 && training; round++) {
        // Snapshot params for delta computation
        let snap;
        try {
          snap = JSON.parse(await wasm.wasm_snapshot_params());
        } catch (e) {
          self.postMessage({ type: 'error', msg: `wasm_snapshot_params crashed: ${e.message}\n${e.stack}` });
          training = false; break;
        }
        if (snap.error) { self.postMessage({ type: 'error', msg: snap.error }); break; }

        const t0 = performance.now();
        for (let i = 0; i < stepsPerRound && training; i++) {
          const batch = getNextBatch(batchSize, seqLen);
          if (!batch) { self.postMessage({ type: 'error', msg: 'No data' }); training = false; break; }

          const lr = cosineLr(i, warmupSteps, stepsPerRound, lrMax, lrMin);
          let r;
          try {
            r = JSON.parse(await wasm.wasm_train_step(lr, batch, batchSize, seqLen));
          } catch (e) {
            self.postMessage({ type: 'error', msg: `wasm_train_step crashed: ${e.message}\n${e.stack}` });
            training = false; break;
          }
          if (r.error) { self.postMessage({ type: 'error', msg: r.error }); training = false; break; }

          step = r.step;
          lastLoss = r.loss ?? 0;
          totalTokens = r.tokens_processed;

          self.postMessage({ type: 'status', step, loss: lastLoss, tokens: totalTokens, round });
          self.postMessage({ type: 'log', msg: `Step ${i + 1}/${stepsPerRound}: loss=${lastLoss.toFixed(4)}, lr=${lr.toExponential(2)}` });
        }
        const secs = (performance.now() - t0) / 1000;

        if (!training) break;

        // Compute and send delta
        let delta, parsed;
        try {
          delta = await wasm.wasm_compute_delta_json();
          parsed = JSON.parse(delta);
        } catch (e) {
          self.postMessage({ type: 'error', msg: `wasm_compute_delta crashed: ${e.message}` });
          parsed = { error: true };
        }
        if (!parsed.error) {
          self.postMessage({ type: 'delta', data: delta, loss: lastLoss, tokens: totalTokens, secs, round });
        }

        self.postMessage({ type: 'roundComplete', round });

        // Brief yield so main thread can send 'stop' or 'loadCheckpoint'
        await new Promise(r => setTimeout(r, 10));
      }

      training = false;
      wasm.wasm_shutdown();
      self.postMessage({ type: 'stopped' });
      break;
    }

    case 'stop': {
      training = false;
      break;
    }

    case 'getStatus': {
      self.postMessage({ type: 'status', step, loss: lastLoss, tokens: totalTokens, round: 0 });
      break;
    }
  }
};
