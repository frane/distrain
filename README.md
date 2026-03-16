# Distrain

Decentralized LLM training. Anyone with a computer contributes to collaboratively training a large language model. A Raspberry Pi and an H100 participate in the same training run.

Distrain uses a CRDT-based continuous merge protocol. There are no rounds, no quorum requirements, and no synchronization barriers. Nodes train locally at their own pace and push small model deltas to a central storage layer. A stateless coordinator merges them into the master checkpoint using Nesterov SGD.

## Architecture

```
                     +------------------+
                     |   Cloudflare R2  |
                     |  (S3-compatible) |
                     |                  |
                     | - checkpoints    |
                     | - model deltas   |
                     | - accumulator    |
                     | - training data  |
                     +--------+---------+
                              |
              reads/writes    |    reads/writes
           +------------------+------------------+
           |                                     |
   +-------v--------+                   +--------v-------+
   |  Coordinator    |                   |  Node          |  (x N)
   |  (Rust/Axum)    |  <--- HTTP --->   |  (Rust/Burn)   |
   |                 |                   |                |
   | 6 JSON endpoints|                   | Train loop:    |
   | Pure Rust       |                   | 1. Pull ckpt   |
   | aggregation     |                   | 2. Train H steps|
   | (Burn/ndarray)  |                   | 3. Upload delta|
   +--------+--------+                   | 4. Push metadata|
            |                            +--------+-------+
    +-------v---------+                           |
    | aggregation.rs  |        +------------------+------------------+
    | Weighted average |        |         |          |         |
    | + Nesterov SGD  |        | CLI     | Desktop  | Browser | Mobile
    +-----------------+        | (Rust)  | (Tauri)  | (WASM)  | (FFI)
                               +------------------+------------------+
```

**Key design decisions:**

- All persistent state lives in R2 (S3-compatible). The coordinator is stateless.
- Nodes upload compressed deltas directly to R2, then POST only the metadata key to the coordinator. The coordinator never handles large payloads.
- Delta contributions are weighted by `(inner_steps / 50) * staleness_decay^staleness`. Fast devices do more steps per push and get proportionally more influence.
- The CRDT accumulator guarantees that concurrent coordinator instances converge to the same state.

## Repository Structure

```
distrain/
+-- coordinator/                  # Axum HTTP server + in-process Rust aggregation
|   +-- src/
|   +-- Dockerfile
+-- core/                         # Shared Rust libraries
|   +-- model/                    # Transformer, config, presets, compression, checkpointing
|   +-- shared/                   # Types, S3 paths, storage client, config
+-- node/                         # All node platforms
|   +-- cli/                      # Native CLI node (Linux/macOS/Windows)
|   +-- desktop/                  # Tauri desktop app (macOS/Windows/Linux)
|   |   +-- src-tauri/            # Rust backend (burn-wgpu GPU training)
|   |   +-- frontend/             # HTML/JS/CSS (from shared UI)
|   +-- browser/                  # Browser node
|   |   +-- wasm/                 # Rust WASM crate (wasm-pack)
|   |   +-- web/                  # HTML/JS + WASM pkg
|   +-- ffi/                      # C FFI crate for mobile
|   +-- ios/                      # iOS app shell (Swift)
|   +-- android/                  # Android app shell (Kotlin)
|   +-- ui/                       # Shared UI (HTML/CSS/JS)
+-- docker/                       # Docker Compose + monitoring (Prometheus, Grafana)
+-- scripts/                      # Python tooling (data prep, eval, simulation)
+-- Cargo.toml                    # Rust workspace
+-- Justfile                      # Build commands per platform
+-- pyproject.toml                # Python tooling config
```

## Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs))
- [just](https://github.com/casey/just) (install: `cargo install just`)
- Docker (for MinIO / local development)
- Python 3.11+ (optional — only needed for data preparation and post-training scripts)

A GPU is optional. CPU and Apple Silicon (Metal via wgpu) work for development and small models.

## Quick Start

```bash
# 1. Build everything
just build-all

# 2. Start MinIO (local S3)
just stack-up

# 3. Bootstrap v0 checkpoint (pure Rust — no Python needed)
just bootstrap

# 4. Start coordinator
just run-coordinator

# 5. Start a training node
just run-node
```

That's it. The node registers with the coordinator, calibrates GPU speed, downloads training data, and starts training. Loss should drop within seconds.

### Multiple node types

```bash
# Terminal 1: Native CLI node (fastest — compiled Rust + GPU)
just run-node

# Terminal 2: Desktop GUI (Tauri — same GPU backend)
just dev-desktop

# Terminal 3: Browser node (WebAssembly)
just serve-browser
# Open http://localhost:8080, enter coordinator URL, click Start Training

# Terminal 4: More CLI nodes
just run-node config=node2.toml
```

All node types push deltas to the same coordinator and contribute to the same checkpoint. Nodes only need the coordinator URL — storage configuration is provided by the coordinator during registration.

## Justfile Commands

Run `just` to see all available commands:

| Command | Description |
|---------|-------------|
| `just test` | Run all Rust tests |
| `just lint` | Run clippy |
| `just check` | Type-check workspace |
| **Coordinator** | |
| `just build-coordinator` | Build coordinator release binary |
| `just run-coordinator` | Run coordinator |
| `just docker-coordinator` | Build coordinator Docker image |
| **Node CLI** | |
| `just build-node` | Build node release binary |
| `just run-node [config]` | Start training (default: `node.toml`) |
| `just bootstrap [config] [preset]` | Create initial checkpoint (default: tiny) |
| `just docker-node` | Build node Docker image |
| **Desktop** | |
| `just dev-desktop` | Run Tauri desktop app in dev mode |
| `just build-desktop` | Build Tauri desktop release |
| **Browser** | |
| `just build-wasm` | Build WASM module |
| `just serve-browser [port]` | Serve browser UI (default: 8080) |
| **Mobile** | |
| `just build-ffi` | Build FFI crate for mobile |
| `just build-ios` | Build iOS framework |
| `just build-android` | Build Android library |
| **Docker** | |
| `just stack-up` | Start full local stack (MinIO + monitoring) |
| `just stack-down` | Stop local stack |
| `just stack-logs` | Tail stack logs |
| **Data** | |
| `just prepare-data [dataset]` | Download and shard training data |
| `just prepare-test` | Create small test shards |

## Components

### Model

Decoder-only Transformer combining Llama 3, Mistral, Qwen 2.5, and DeepSeek design choices. Implemented entirely in Rust using the [Burn](https://burn.dev) ML framework.

| Component | Implementation |
|-----------|----------------|
| Attention | Grouped-Query Attention + RoPE (theta=500k) |
| FFN | SwiGLU (gate + up + down projections) |
| Normalization | RMSNorm (pre-norm) |
| Embedding | Tied input/output with learned output scaling |
| QKV bias | Yes (Qwen 2.5 style) |
| Initialization | Per-layer scaled init (DeepSeek style) |

Size presets:

| Preset | Parameters | Hidden | Layers | Heads | KV Heads |
|--------|-----------|--------|--------|-------|----------|
| `micro-test` | ~1M | 64 | 2 | 4 | 2 |
| `tiny` / `125m` | 100M | 768 | 12 | 12 | 4 |
| `small` / `1b` | 1.1B | 2048 | 24 | 16 | 4 |
| `medium` / `7b` | 5.8B | 4096 | 32 | 32 | 8 |
| `large` / `13b` | 11.5B | 5120 | 40 | 40 | 8 |

### Coordinator

Stateless Axum HTTP server. All persistent state lives in R2.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/nodes/register` | Register a node. Returns node ID, API key, and storage config. |
| POST | `/delta` | Push delta metadata. Triggers aggregation when threshold met. |
| PUT | `/upload/*key` | Proxy upload to R2 (used by browser nodes for CORS) |
| GET | `/download/*key` | Proxy download from R2 (used by browser nodes for CORS) |
| GET | `/checkpoint/latest` | Current checkpoint version + S3 key |
| GET | `/status` | Training status (public) |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

**Configuration (environment variables):**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Listen port |
| `HOST` | `0.0.0.0` | Listen address |
| `R2_ENDPOINT` | `http://localhost:9000` | S3-compatible endpoint |
| `R2_BUCKET` | `distrain-training` | Bucket name |
| `R2_ACCESS_KEY_ID` | `minioadmin` | Access key |
| `R2_SECRET_ACCESS_KEY` | `minioadmin` | Secret key |
| `MIN_CONTRIBUTIONS` | `4` | Deltas needed before checkpoint production |

When the accumulator reaches `MIN_CONTRIBUTIONS`, the coordinator runs aggregation: downloads all pending deltas from R2, computes a weighted average, applies Nesterov SGD, and uploads the new checkpoint. Pure Rust — no Python, no external processes.

### Node Client

Single binary. No Python. No PyTorch. Trains on any GPU or CPU using Burn.

```bash
# Start training
distrain-node start --config node.toml

# Bootstrap initial checkpoint
distrain-node bootstrap --config node.toml --preset tiny

# Benchmark this device
distrain-node benchmark --config node.toml
```

**`node.toml` configuration:**

```toml
coordinator_url = "http://localhost:8000"
api_key = ""
gpu_device = 0            # -1 for CPU, 0+ for GPU index
target_push_interval_secs = 60.0
min_inner_steps = 10
max_inner_steps = 500
cache_dir = "~/.distrain/cache"
max_cache_gb = 100

[storage]
endpoint = "http://localhost:9000"
bucket = "distrain-training"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
```

**Node training loop:**

1. Register with coordinator (receives node ID + storage config)
2. Calibrate GPU speed, compute optimal H_mini
3. Download latest checkpoint from R2
4. Compute shard assignment: `compute_shard_assignment(node_id, version, total_shards, shards_per_node)` — deterministic, no coordinator involvement, different nodes get different data
5. Load assigned data shards
6. Train H_mini steps (Burn — forward, backward, AdamW)
7. Compute delta and compress (top-k + INT8 + zstd)
8. Upload compressed delta to R2
9. POST delta metadata to coordinator
10. If checkpoint version advanced, download new checkpoint + recompute shard assignment
11. Repeat from step 6

The node never waits for other nodes. It trains continuously. Each node trains on different data — `compute_shard_assignment` uses `hash(node_id, checkpoint_version)` as a deterministic seed, so shard selection is reproducible and verifiable without coordination.

### Shared UI

All graphical node platforms (desktop, browser, mobile) share the same HTML/CSS/JS UI from `node/ui/`. Each platform provides a thin adapter implementing:

```javascript
{
  platformName: 'browser' | 'desktop' | 'ios' | 'android',
  startTraining()  → Promise<string>,
  stopTraining()   → Promise<string>,
  getStatus()      → Promise<NodeStatus>,
  getStats()       → Promise<TrainingStats>,
  getLogs()        → Promise<string[]>,
}
```

Run `just sync-ui` to copy shared UI files to platform directories. Build targets (`build-desktop`, `build-wasm`, etc.) do this automatically.

### Desktop Node (Tauri)

Native desktop app using Tauri 2. Same GPU training backend as the CLI node (burn-wgpu: Metal on macOS, Vulkan on Linux/Windows).

```bash
just dev-desktop    # Development with hot-reload
just build-desktop  # Release build
```

### Browser Node (WebAssembly)

The same Burn model compiled to WebAssembly. Uses WebGPU when available (Chrome 113+), falls back to CPU. Training runs in a Web Worker to keep the UI responsive.

```bash
just build-wasm           # Build WASM module
just serve-browser        # Serve at localhost:8080
```

Open the browser, enter the coordinator URL, and click Start Training. The browser node only needs the coordinator URL — it receives storage configuration automatically during registration. Browser nodes route all S3 traffic through the coordinator's upload/download proxy endpoints (CORS).

### Mobile (iOS/Android)

Shared C FFI crate (`node/ffi/`) with native app shells:

```bash
just build-ffi       # Build FFI crate
just build-ios       # Build iOS framework
just build-android   # Build Android library
```

## Bootstrapping

Create the initial (v0) checkpoint. This is pure Rust — no Python, no PyTorch:

```bash
# Default: tiny preset (100M params)
just bootstrap

# Specific preset
just bootstrap node.toml small

# Direct cargo
cargo run -p distrain-node -- bootstrap --config node.toml --preset 7b
```

This initializes a model with the given preset, serializes it to safetensors, uploads it to R2, and creates the initial accumulator state. The system is then ready for nodes to start training.

## Data Preparation

Prepare tokenized binary shards for training (requires Python):

```bash
# Install Python dependencies
pip install -e ".[data]"

# Download and shard FineWeb-Edu (1T tokens)
just prepare-data fineweb-edu-10bt

# Small test shards
just prepare-test
```

Shard format: flat arrays of uint16 token IDs (little-endian), ~200MB per shard.

## Delta Compression

Model deltas are compressed to fit on residential internet connections:

1. **Top-k sparsification**: Keep top 0.1-1% of elements by magnitude
2. **Error feedback**: Dropped elements accumulate locally and re-enter the top-k set on subsequent pushes (no information permanently lost)
3. **INT8 quantization**: Reduce precision
4. **zstd compression**: Sorted-index encoding + zstd

For a 7B model: ~14 GB raw (BF16) -> ~100-500 MB per push.

## Device Calibration

Every device auto-calibrates its inner steps per push (H_mini) to target ~60 second push intervals:

```
Device              Speed           H_mini     Push interval    Weight
H100                0.16 sec/step   375        ~60s             7.5
RTX 4090            0.33 sec/step   181        ~60s             3.6
M2 MacBook          2.0  sec/step   30         ~60s             0.6
CPU server          10   sec/step   10         ~100s            0.2
Raspberry Pi        60   sec/step   10         ~600s            0.04
```

All devices contribute. Weight = `(inner_steps / 50) * data_quality * 0.9^staleness`.

## Monitoring

The coordinator exposes Prometheus metrics at `/metrics`. The Docker Compose stack includes Prometheus and Grafana:

```bash
just stack-up
# Grafana: http://localhost:3000 (admin / distrain)
# Prometheus: http://localhost:9090
# MinIO Console: http://localhost:9001 (minioadmin / minioadmin)
```

## Protocol

```
Node (Rust + Burn):
  1. Pull latest checkpoint from R2
  2. Compute shard assignment: hash(node_id, version) → deterministic shard list
  3. Load assigned data shards (different nodes see different data)
  4. Train H_mini steps with AdamW on assigned data
  5. Compute delta = snapshot_before - params_after
  6. Compress: top-k + error feedback + INT8 + zstd
  7. Upload compressed delta to R2
  8. POST /delta with {delta_key, inner_steps, seq_num, ...}

Coordinator (Rust, on POST /delta):
  1. Validate: staleness check, seq_num dedup
  2. Compute weight: (inner_steps / 50) * 0.9^staleness
  3. Add to CRDT accumulator
  4. If contributions >= min_contributions:
     a. Download all pending deltas from R2
     b. Decompress, validate, compute weighted average
     c. Apply Nesterov SGD: v = mu*v + delta; theta -= lr*(mu*v + delta)
     d. Save new checkpoint + velocity to R2
     e. Reset accumulator
```

## S3 Path Layout

All paths defined in `core/shared/src/paths.rs`:

```
distrain-training/                           # bucket
  checkpoints/v{N}/model.safetensors         # model weights
  checkpoints/v{N}/metadata.json             # aggregation metadata
  optimizer_state/v{N}/velocity.safetensors   # Nesterov momentum
  deltas/v{N}/{node_id}_{seq}.delta.zst      # compressed deltas
  accumulator/current.json                   # CRDT accumulator state
  state/node_registry.json                   # registered nodes
  data/shard_{NNNN}.bin                      # tokenized data shards
  data/manifest.json                         # shard manifest
```

## Tests

```bash
# All Rust tests
just test

# Lint
just lint

# Python tests (optional — post-training tools)
pip install -e ".[dev]"
pytest scripts/ -v
```

## Docker

```bash
# Build images
just docker-coordinator
just docker-node

# Full local stack (MinIO + Prometheus + Grafana)
just stack-up
just stack-down
```

Both images use multi-stage builds: Rust compilation in a builder stage, then minimal Debian runtime. No Python, no PyTorch — pure Rust binaries.
