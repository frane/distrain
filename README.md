# Distrain

Distributed LLM training on consumer hardware. Like SETI@home but for training language models.

Anyone with a computer (GPU, CPU, laptop, whatever) can contribute to training an LLM from scratch. Your device trains locally, compresses the weight update, and pushes it to a coordinator that merges everything into a shared checkpoint. No synchronization, no waiting. A fast GPU pushes every few minutes, a slow laptop pushes every half hour. Both contribute.

Written entirely in Rust. Single binary, no Python runtime, no PyTorch. Runs on CUDA, Metal, Vulkan, CPU, and compiles to WebAssembly for in-browser training.

## Does it actually work?

Yes. We trained a 125M parameter model on 3 consumer devices over a LAN and observed convergence from random init (loss 700) to loss 8.8 on a single-GPU baseline, with the distributed system tracking close behind.

We also trained a 1B parameter model, with loss dropping from 1,669 to 35 over 87 checkpoints. Each push compresses a 4.3 GB model delta down to about 88 MB (using 20% top-k sparsification with per-tensor adaptive budget allocation).

The system includes self-optimizing feedback loops: adaptive inner/outer learning rates, loss spike detection with automatic round abort, delta norm clipping, and bandwidth-aware compression. The protocol tunes its own hyperparameters from observed training dynamics.

## The protocol

There's no AllReduce, no parameter server, no synchronous rounds. Deltas can arrive in any order and the result is the same.

Each node:
1. Downloads the latest checkpoint
2. Trains for a configurable number of steps (auto-calibrated per device)
3. Compresses the delta: per-tensor adaptive top-k sparsification (5-60% by magnitude), error feedback (dropped values accumulate and re-enter future pushes), INT8 quantization, and zstd
4. Uploads to object storage, pings the coordinator
5. Coordinator merges deltas with staleness-weighted averaging (outer LR close to 1.0)
6. New checkpoint appears, repeat

Stale deltas (computed against old checkpoints) get exponentially less weight: `0.9^staleness`. Deltas more than 10 versions behind are discarded. Slow devices contribute proportionally without holding anything back.

Per-step heartbeats let the coordinator track node liveness and tell stale nodes to abort their current round and pull the new checkpoint.

## Running it

### Local development

```bash
cargo build --release -p distrain-coordinator -p distrain-node

# Start MinIO (local S3)
docker compose -f docker/docker-compose.yml up -d minio

# Prepare training data (needs Python + HuggingFace libraries)
pip install datasets tokenizers numpy
python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb --upload

# Bootstrap a model
./target/release/distrain-node bootstrap --config node.toml --preset tiny

# Run coordinator
RUST_LOG=info ./target/release/coordinator

# Run node (separate terminal)
./target/release/distrain-node start --config node.toml
```

### Cloud deployment (Kubernetes)

The coordinator and MinIO deploy to any k8s cluster:

```bash
kubectl apply -k k8s/overlays/do
```

Training nodes connect from anywhere (cloud GPUs, consumer devices, laptops) by pointing at the coordinator's external IP in `node.toml`.

### Docker

```bash
# Coordinator
docker pull ghcr.io/frane/distrain/coordinator:latest

# Node (CPU/Metal)
docker pull ghcr.io/frane/distrain/node:latest

# Node with NVIDIA GPU support (Vulkan)
docker pull ghcr.io/frane/distrain/node-gpu:latest
```

### Single-GPU baseline (for comparison)

```bash
distrain-node baseline <checkpoint> --data-dir <shards> --steps 2000 --output baseline.jsonl
```

Trains the same model with the same data and hyperparameters but without the distributed protocol. For measuring convergence penalty.

## What's here

```
coordinator/        HTTP server (Axum) + aggregation logic
core/model/         Transformer, compression, checkpointing
core/shared/        Storage client, config, shared types
node/cli/           The training node
node/desktop/       Tauri desktop app (experimental)
node/browser/       WebAssembly version (experimental)
node/ffi/           C FFI for mobile (experimental)
scripts/            Data prep, eval, benchmarks
docker/             Local dev stack (MinIO, Prometheus, Grafana)
k8s/                Kubernetes manifests (kustomize)
```

## Self-optimizing protocol

The system tunes its own parameters from observed training dynamics:

- **Adaptive inner LR**: node reduces learning rate when loss variance is high, restores when stable
- **Continuous outer LR**: smooth log-linear schedule approaching 1.0 as training stabilizes
- **Loss spike detection**: aborts round and skips push if loss diverges mid-training
- **Delta norm clipping**: scales down outlier deltas to 3x running average (preserves direction)
- **Per-tensor adaptive top-k**: allocates compression budget proportionally to gradient norm per tensor
- **Bandwidth measurement**: logs upload speed for future adaptive compression

## Model

Standard decoder-only transformer (GQA + RoPE + SwiGLU + RMSNorm). Implemented in Rust via [Burn](https://burn.dev). Presets from 1M to 13B parameters.

The node auto-detects GPU vs CPU, picks the faster backend, calibrates batch size (retries on OOM), and figures out how many steps to run per push.

## Related work

- [DiLoCo](https://arxiv.org/abs/2311.08105) (Google, 2023) introduced the inner-outer optimization idea that inspired this project
- [INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) (Prime Intellect) trained a 10B model across continents, but relies on datacenter GPUs and synchronous outer steps
- [Psyche](https://nousresearch.com/nous-psyche/) (Nous Research) is a decentralized training network that supports consumer GPUs but still requires synchronization
- [Hivemind](https://github.com/learning-at-home/hivemind) (Together.ai) is a PyTorch library for decentralized training, mostly used for inference

Distrain differs in three ways: fully asynchronous merge with no synchronization points, a pure Rust single binary with zero runtime dependencies, and support for truly heterogeneous hardware including CPUs and browsers via WASM.

## License

MIT
