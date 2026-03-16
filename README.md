# Distrain — SETI@home for AI Training

Train LLMs from scratch using idle computers around the world. A MacBook, a gaming PC, and a cloud GPU all contribute to the same training run — no datacenter required.

Distrain uses a **CRDT-based asynchronous merge protocol** that eliminates synchronization barriers. There are no rounds, no quorum, no waiting. Fast devices push every few seconds, slow devices push every few minutes. The math handles the heterogeneity.

**Everything is Rust.** Single static binary. No Python. No PyTorch. No dependencies. Runs on CUDA, Metal, Vulkan, CPU — and compiles to WebAssembly for in-browser training.

## Results

We trained a **125M parameter model to convergence** using 3 consumer devices over a LAN:

| Device | Type | Speed |
|--------|------|-------|
| M2 Pro 32GB | GPU (Metal) | ~1,024 tok/s |
| MacBook Air M2 24GB | GPU (Metal) | ~800 tok/s |
| Intel i7 MacBook Pro | CPU only | ~100 tok/s |

**Loss: 674 (random init) → 6.3 (converged)** over 2,928 checkpoints.

A **1B parameter model** is currently training on the same hardware, with loss dropping from 1,669 → 65 over 48 checkpoints. Delta compression reduces each push from 4.5 GB to ~25 MB (175x compression).

<!-- TODO: Add loss curve image -->

## How It Works

```
Every device, regardless of speed:

1. Downloads the latest model checkpoint
2. Trains locally for a while (H100: hundreds of steps, laptop: 10 steps)
3. Compresses the weight delta (top-k sparsification + error feedback + INT8 + zstd)
4. Uploads the compressed delta (~25-500 MB for a 7B model)
5. The coordinator merges all deltas via weighted average (CRDT — commutative, associative)
6. New checkpoint produced → everyone pulls it and repeats

No synchronization barriers. No waiting for the slowest node.
```

### Key properties

- **CRDT merge**: Deltas can arrive in any order, at any time, and produce the same result. No coordination needed.
- **Staleness weighting**: Deltas computed against old checkpoints are weighted down exponentially (`0.9^staleness`). Slow devices naturally get proportionally lower influence.
- **Adaptive calibration**: Each device auto-tunes its training steps per push to target ~60 second intervals.
- **Error feedback**: Values dropped by top-k sparsification accumulate locally and re-enter future pushes. No gradient information is permanently lost.
- **Loss-based outer learning rate**: The coordinator adjusts its learning rate based on absolute loss relative to `ln(vocab_size)`, adapting automatically to training phase.

### Compression pipeline

A 7B model delta is 14 GB raw in BF16. To get to residential internet sizes:

1. **Top-k sparsification** — keep top 0.1-1% of elements by magnitude (adaptive based on loss)
2. **Error feedback** — dropped elements accumulate locally, re-enter top-k next push
3. **INT8 quantization** — reduce precision of kept values
4. **Sorted-index encoding + zstd** — final compression

Result: **~100-500 MB per push for 7B**, depending on training phase.

## Quick Start

```bash
# Build coordinator and node
cargo build --release -p distrain-coordinator -p distrain-node

# Start MinIO (local S3-compatible storage)
docker compose -f docker/docker-compose.yml up -d minio

# Prepare training data (requires Python)
pip install -e ".[data]"
python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb --upload

# Bootstrap initial checkpoint
./target/release/distrain-node bootstrap --config node.toml --preset tiny

# Start coordinator
RUST_LOG=info ./target/release/coordinator

# Start training (in another terminal)
./target/release/distrain-node start --config node.toml
```

The node registers with the coordinator, auto-detects GPU vs CPU, calibrates training speed, downloads data shards, and starts training. Loss should drop within the first round.

### Adding more nodes

Any machine on the network can join by pointing at the coordinator:

```toml
# node.toml
coordinator_url = "http://<coordinator-ip>:8000"

[storage]
endpoint = "http://<coordinator-ip>:9000"
bucket = "distrain-training"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
```

```bash
./distrain-node start --config node.toml
```

Different nodes automatically train on different data via deterministic shard assignment (`hash(node_id, checkpoint_version)` as seed). No coordinator involvement needed.

## Architecture

```
                    +-----------------+
                    |  Object Storage |
                    |  (R2 / MinIO)   |
                    |                 |
                    | checkpoints/    |
                    | deltas/         |
                    | data/           |
                    +--------+--------+
                             |
             reads/writes    |    reads/writes
          +------------------+------------------+
          |                                     |
  +-------v--------+                   +--------v-------+
  |  Coordinator   |  <--- HTTP --->   |     Node       |  (x N)
  |  (Rust/Axum)   |                   |  (Rust/Burn)   |
  |                |                   |                |
  | 5 endpoints    |                   | Train loop:    |
  | CRDT merge     |                   | 1. Pull ckpt   |
  | Nesterov SGD   |                   | 2. Train H steps|
  | Housekeeping   |                   | 3. Compress delta|
  +----------------+                   | 4. Upload + push |
                                       +----------------+
```

### Coordinator

Stateless Rust HTTP server (Axum). All persistent state lives in object storage.

| Endpoint | Description |
|----------|-------------|
| `POST /nodes/register` | Register device, get node ID and config |
| `POST /delta` | Push delta metadata, triggers aggregation |
| `GET /checkpoint/latest` | Current checkpoint version + URL |
| `GET /status` | Training progress (public) |
| `GET /health` | Health check |

When enough deltas accumulate, the coordinator downloads them from storage, computes a weighted average, applies a Nesterov momentum step, and uploads the new checkpoint. Pure Rust aggregation via Burn — no Python.

### Node

Single static binary. Trains on any hardware using [Burn](https://burn.dev):

| Backend | Hardware |
|---------|----------|
| burn-wgpu | NVIDIA (Vulkan), Apple Silicon (Metal), AMD (Vulkan) |
| burn-cuda | NVIDIA (CUDA) |
| burn-ndarray | Any CPU |

Auto-detects the best backend, calibrates batch size and steps per push, handles OOM recovery with automatic gradient accumulation.

### Model

Decoder-only Transformer (Llama/Mistral/Qwen-style):

| Component | Choice |
|-----------|--------|
| Attention | GQA + RoPE (theta=500k) |
| FFN | SwiGLU |
| Norm | RMSNorm (pre-norm) |
| Embedding | Tied input/output with learned scaling |
| Precision | BF16 weights, FP32 optimizer |

Size presets:

| Preset | Parameters | Hidden | Layers | Heads | KV Heads |
|--------|-----------|--------|--------|-------|----------|
| `micro-test` | ~1M | 64 | 2 | 4 | 2 |
| `tiny` | 125M | 768 | 12 | 12 | 4 |
| `small` | 1.1B | 2048 | 24 | 16 | 4 |
| `medium` | 7B | 4096 | 32 | 32 | 8 |
| `large` | 13B | 5120 | 40 | 40 | 8 |

## Device Heterogeneity

Every device auto-calibrates to target ~60 second push intervals:

```
Device              Speed           Steps/push   Weight
H100                0.16 sec/step   375          7.5
RTX 4090            0.33 sec/step   181          3.6
M2 MacBook          2.0  sec/step   30           0.6
CPU server          10   sec/step   10           0.2
Raspberry Pi        60   sec/step   10           0.04 (stale)
```

Weight = `(inner_steps / 50) * 0.9^staleness`. All devices contribute proportionally.

## Storage Layout

```
distrain-training/
  checkpoints/v{N}/model.safetensors
  checkpoints/v{N}/metadata.json
  optimizer_state/v{N}/velocity.safetensors
  deltas/v{N}/{node_id}_{seq}.delta.zst
  accumulator/current.json
  state/coordinator.json
  state/node_registry.json
  data/shard_{NNNN}.bin
  data/manifest.json
```

## Related Work

Distrain builds on ideas from:

- **[DiLoCo](https://arxiv.org/abs/2311.08105)** (Google DeepMind, 2023) — inner-outer optimization for distributed training
- **[OpenDiLoCo / INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1)** (Prime Intellect) — 10B model trained across 3 continents using datacenter GPUs
- **[Psyche](https://nousresearch.com/nous-psyche/)** (Nous Research) — decentralized training network with consumer GPU support
- **[Hivemind](https://github.com/learning-at-home/hivemind)** (Together.ai) — PyTorch library for decentralized training
- **Deep Gradient Compression** (Lin et al., ICLR 2018) — top-k sparsification
- **Error Feedback** (Stich et al., NeurIPS 2018; Karimireddy et al., ICML 2019)

Key differences from existing approaches:
- **Fully asynchronous** — no synchronous outer optimization step (DiLoCo/Psyche synchronize)
- **Zero dependencies** — single Rust binary, no Python runtime, no PyTorch
- **True consumer hardware** — CPUs, integrated GPUs, browsers via WASM (not just datacenter GPUs)

## Repository Structure

```
coordinator/          Axum HTTP server + Rust aggregation
core/
  model/              Transformer, presets, compression, checkpointing
  shared/             Storage client, paths, config, types
node/
  cli/                Native CLI training node (primary)
  desktop/            Tauri desktop app (experimental)
  browser/wasm/       WebAssembly node (experimental)
  ffi/                C FFI for mobile (experimental)
  ios/                iOS app shell (experimental)
  android/            Android app shell (experimental)
scripts/              Data preparation, eval, benchmarks
docker/               Docker Compose + monitoring
```

## Building

```bash
# Prerequisites: Rust 1.75+, just (cargo install just)

# Build everything
just build-all

# Run tests
just test

# Lint
just lint
```

## License

MIT
