# Distrain

Distributed LLM training on consumer hardware. Like SETI@home but for training language models.

Anyone with a computer (GPU, CPU, laptop, whatever) can contribute to training an LLM from scratch. Your device trains locally, compresses the weight update, and pushes it to a coordinator that merges everything into a shared checkpoint. No synchronization, no waiting. A fast GPU pushes every few minutes, a slow laptop pushes every half hour. Both contribute.

Written entirely in Rust. Single binary, no Python runtime, no PyTorch. Runs on CUDA, Metal, Vulkan, CPU, and compiles to WebAssembly for in-browser training.

## Does it actually work?

Yes. We systematically measured the overhead of distributed training vs a single-GPU baseline, optimizing one variable at a time across six experiment curves:

| Curve | Plateau Loss | Gap to Baseline | Key Optimization |
|-------|-------------|-----------------|------------------|
| Baseline (A) | 4.8 | — | Single GPU, no protocol |
| B | 6.7 | +1.9 | Naive distributed (first attempt) |
| C | 6.4 | +1.6 | Outer LR tuning, top-k 40-65% |
| D | 6.0 | +1.2 | Patience triggers, error buffer 90% |
| E | 5.8 | +1.0 | Continuous training (GPU never idles) |
| F | **in progress** | **ahead of BL** | Auto-tuning, raw deltas, LR scaling |

**Curve F beats the single-GPU baseline in token efficiency.** At 2M tokens: distributed loss = 11.1 vs baseline loss = 21.6 (49% better). This is possible because (a) three nodes see 3x more diverse training data per checkpoint, (b) learning rate scales with the effective batch size, and (c) near-raw deltas preserve all gradient signal.

The tradeoff is bandwidth: raw deltas for a 125M model are ~400MB per push. The system adapts automatically — datacenter nodes send raw (best quality), residential nodes compress to fit their connection (lower quality but accessible from anywhere).

## The protocol

There's no AllReduce, no parameter server, no synchronous rounds. Deltas can arrive in any order and the result is the same.

Each node:
1. Downloads the latest checkpoint
2. Trains continuously (GPU never idles between rounds)
3. Compresses the delta based on measured upload bandwidth — raw if it fits, top-k sparsification + zstd if not. Error feedback ensures dropped values re-enter on the next push.
4. Uploads to object storage, pings the coordinator
5. Coordinator merges deltas with staleness-weighted averaging (outer LR = 1.0)
6. New checkpoint appears, training continues without pause

Stale deltas (computed against old checkpoints) get exponentially less weight: `0.9^staleness`. Slow devices contribute proportionally without holding anything back.

## Everything auto-tunes

The node discovers its own hardware and configures itself:

| Parameter | How it's determined |
|-----------|-------------------|
| Batch size | Computed from GPU VRAM and model architecture. OOM → halve and retry. |
| Learning rate | Scales linearly with batch size (reference: batch=4 at lr=3e-4) |
| Steps per push (H_mini) | Measured from actual upload time. Target: ~10s uploads. |
| Compression | Bandwidth-adaptive. Tries raw first, falls back to top-k based on upload budget. |
| Shards in memory | Computed from available RAM after model overhead |
| Contribution weight | Tokens processed × staleness decay — no magic numbers |
| Min contributions | Auto-computed from active node count (at least half must contribute) |

No hardcoded device-specific values. A Raspberry Pi and an H100 participate in the same training run.

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

### Cloud deployment (Docker)

```bash
# Node with NVIDIA GPU (CUDA)
docker pull ghcr.io/frane/distrain/node-cuda:latest

# Run with auto-configuration
docker run --gpus all \
  -e COORDINATOR_URL=http://your-coordinator:8000 \
  -e S3_ENDPOINT=http://your-minio:9000 \
  -e S3_ACCESS_KEY=distrain \
  -e S3_SECRET_KEY=secret \
  -e S3_BUCKET=distrain-training \
  ghcr.io/frane/distrain/node-cuda:latest
```

The node auto-detects GPU, estimates batch size from VRAM, starts training immediately. No manual configuration needed.

### Single-GPU baseline (for comparison)

```bash
distrain-node baseline <checkpoint> --data-dir <shards> --steps 2000 --output baseline.jsonl
```

Trains the same model with the same data and hyperparameters but without the distributed protocol.

## What's here

```
coordinator/        HTTP server (Axum) + aggregation (burn tensors)
core/model/         Transformer, compression, checkpointing (Burn)
core/shared/        Storage client, config, shared types
node/cli/           The training node (continuous training, auto-tuning)
node/desktop/       Tauri desktop app (experimental)
node/browser/       WebAssembly version (experimental)
node/ffi/           C FFI for mobile (experimental)
scripts/            Data prep, eval, post-training (SFT/DPO)
docker/             Local dev stack (MinIO, Prometheus, Grafana)
```

## Model

Standard decoder-only transformer (GQA + RoPE + SwiGLU + RMSNorm). Implemented in Rust via [Burn](https://burn.dev). Presets from 1M to 13B parameters.

## Related work

- [DiLoCo](https://arxiv.org/abs/2311.08105) (Google, 2023) — inner-outer optimization that inspired this project
- [INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) (Prime Intellect) — 10B across continents, datacenter GPUs, synchronous outer steps
- [Psyche](https://nousresearch.com/nous-psyche/) (Nous Research) — decentralized training, consumer GPUs, requires synchronization
- [Hivemind](https://github.com/learning-at-home/hivemind) (Together.ai) — PyTorch library for decentralized training

Distrain differs in three ways: fully asynchronous merge with no synchronization points, a pure Rust single binary with zero runtime dependencies, and support for truly heterogeneous hardware including CPUs and browsers via WASM.

## License

MIT
