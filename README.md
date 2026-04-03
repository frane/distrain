# Distrain

Asynchronous distributed LLM training with zero synchronization. A research prototype exploring whether volunteer computing can train language models.

Multiple GPUs train the same model without coordination. Each device trains locally, pushes weight updates to a coordinator, and the coordinator merges them. No AllReduce, no synchronous rounds, no waiting for the slowest node. A datacenter GPU and a gaming PC participate in the same training run — each auto-configures and contributes proportionally.

Written entirely in Rust. Single binary, no Python runtime, no PyTorch. Runs on CUDA, Metal, Vulkan, and CPU.

> **Status: Research preview.** The protocol works and the code runs, but this is not production software. We're looking for collaborators to test on larger models, more nodes, and real-world network conditions. See [Contributing](#contributing) below.

## Results

We trained a 125M parameter transformer across seven experiment curves, measuring the gap between distributed and single-GPU training:

| Curve | Hardware | Plateau Loss | Gap to Baseline | Key Change |
|-------|----------|-------------|-----------------|------------|
| Baseline (A) | 1× RTX 4000 Ada | 4.8 | — | Single GPU, batch=4, lr=3e-4 |
| B | 3× Mac (2 GPU + 1 CPU) | 6.7 | +1.9 | First distributed, heterogeneous consumer hardware |
| C | 3× mixed RTX | 6.4 | +1.6 | Outer LR→1.0, compression tuning |
| D | 3× RTX A4000 | 6.0 | +1.2 | Patience triggers, all nodes contributing |
| E | 3× A40 | 5.8 | +1.0 | Continuous training (GPU never idles) |
| H | 3× A40 | 5.98 | +1.2 | Clean comparison: same hyperparams as baseline |
| G | 3× A40 | 6.4 | +1.6 | Loss-based lr (decays as model converges) |

Validation loss on held-out data matches training loss within 0.4% across all curves, confirming no overfitting.

**Key findings:**

- **The gap is 1.0-1.2 points** between distributed (3 nodes) and single-GPU baseline. This comes from compression loss (~10% signal per round) and checkpoint merge staleness (~30s per merge on CPU).
- **More contributors = better quality.** 3-contribution checkpoints consistently outperform 2-contribution ones.
- **Consumer hardware works.** Curve B trained across a 10× throughput gap (Apple Silicon GPUs + Intel CPU) for 2,928 checkpoints.
- **Compression is tunable.** Top-k retention ranges from 1% (tiny deltas, more signal loss) to 99% (near-raw, best quality). The system auto-selects based on measured upload bandwidth. On fast connections, it sends near-raw deltas for maximum quality.

## The protocol

Each node:
1. Downloads the latest checkpoint
2. Trains continuously (GPU never idles between rounds)
3. Compresses the delta based on measured upload bandwidth — raw if it fits, top-k sparsification + zstd if not. Error feedback ensures no gradient information is permanently lost.
4. Uploads to object storage, notifies the coordinator
5. Coordinator merges deltas with staleness-weighted averaging (outer LR = 1.0)
6. New checkpoint produced, training continues without pause

Stale deltas get exponentially less weight: `0.9^staleness`. The merge is commutative within each accumulation window.

## Everything auto-tunes

| Parameter | How it's determined |
|-----------|-------------------|
| Batch size | Computed from GPU VRAM + model architecture. OOM → halve and retry. |
| Learning rate | Scales linearly with effective batch size |
| Steps per push | Measured from actual upload time |
| Compression | Bandwidth-adaptive: raw if fast, compressed if slow |
| Shards in memory | Computed from available RAM |
| Contribution weight | Tokens processed × staleness decay |
| Min contributions | Auto-computed from active node count |

## Model presets

Training a different model size is just an environment variable:

| Preset | Params | Hidden | Layers | Heads | KV Heads | FFN | VRAM needed |
|--------|--------|--------|--------|-------|----------|-----|-------------|
| `micro-test` | 64K | 64 | 2 | 2 | 2 | 256 | <1 GB |
| `tiny` | 125M | 768 | 12 | 12 | 4 | 2048 | ~8 GB |
| `small` | 1.13B | 2048 | 24 | 16 | 4 | 5504 | ~40 GB |
| `medium` | 7B | 4096 | 32 | 32 | 8 | 11008 | ~80 GB |
| `large` | 13B | 5120 | 40 | 40 | 8 | 13824 | ~160 GB |

Set `PRESET=small` on the coordinator to bootstrap a 1B model. Nodes auto-detect the architecture from the checkpoint — no configuration needed.

## Running it

### Docker deployment (recommended)

Three steps: start coordinator, prepare training data (once), start nodes.

**Step 1: Start the coordinator**

The coordinator image includes MinIO (S3-compatible storage), bootstraps a random v0 checkpoint on first start, and persists all data on the `/workspace` volume.

```bash
docker run --gpus all \
  -e S3_ACCESS_KEY=distrain \
  -e S3_SECRET_KEY=yoursecret \
  -e PRESET=tiny \
  -v distrain-data:/workspace \
  -p 8000:8000 -p 9000:9000 -p 22:22 \
  ghcr.io/frane/distrain/coordinator:latest
```

**Step 2: Prepare training data (once per dataset)**

The coordinator image includes `prepare_data.py` and the Mistral v0.3 tokenizer. SSH or exec into the container and run:

```bash
python3 /scripts/prepare_data.py fineweb-edu-10bt \
  --output-dir /tmp/data \
  --upload \
  --s3-endpoint http://localhost:9000 \
  --s3-bucket distrain-training \
  --s3-access-key distrain \
  --s3-secret-key yoursecret
```

This downloads FineWeb-Edu from HuggingFace (~10B tokens), tokenizes it (~1 hour), and uploads 1,102 shards to MinIO. You only need to do this once.

**Step 3: Start training nodes**

```bash
docker run --gpus all \
  -e COORDINATOR_URL=http://your-coordinator:8000 \
  -e S3_ENDPOINT=http://your-coordinator:9000 \
  -e S3_ACCESS_KEY=distrain \
  -e S3_SECRET_KEY=yoursecret \
  -e S3_BUCKET=distrain-training \
  ghcr.io/frane/distrain/node-cuda:latest
```

Each node auto-detects its GPU, computes optimal batch size, and starts training immediately. Start as many nodes as you want.

### Local development

```bash
cargo build --release -p distrain-coordinator -p distrain-node

# Start MinIO (local S3)
docker compose -f docker/docker-compose.yml up -d minio

# Prepare training data
pip install datasets tokenizers numpy tqdm
python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb --upload

# Bootstrap a model
./target/release/distrain-node bootstrap --config node.toml --preset tiny

# Run coordinator
RUST_LOG=info ./target/release/coordinator

# Run node (separate terminal)
./target/release/distrain-node start --config node.toml
```

## What's here

```
coordinator/        HTTP server (Axum) + aggregation (burn tensors)
core/model/         Transformer, compression, checkpointing (Burn)
core/shared/        Storage client, config, shared types
node/cli/           Training node (continuous training, auto-tuning)
node/desktop/       Tauri desktop app (experimental)
node/browser/       WebAssembly version (experimental)
node/ffi/           C FFI for mobile (experimental)
scripts/            Data prep, eval, post-training
docker/             Local dev stack (MinIO, Prometheus, Grafana)
```

## Open questions and known limitations

These are the real problems we haven't solved yet:

- **1.0-1.2 point quality gap.** Distributed training plateaus above single-GPU. The gap comes from compression loss (~10% of gradient signal per round) and merge staleness (~30s of training against an outdated checkpoint). GPU-accelerated aggregation and delta streaming (more frequent, smaller pushes) are promising directions.

- **Delta size vs quality tradeoff.** Raw deltas give the best quality but are large (300MB for 125M, ~14GB for 7B). Top-k compression reduces size but loses signal. Low-rank compression doesn't work for pre-training (deltas are full-rank). The system auto-adapts compression to bandwidth, but residential internet users will always get worse quality than datacenter nodes.

- **Only validated at small scale.** 125M parameters, 3 nodes. The protocol is designed for 7B+ models with 50-1000 nodes, but staleness handling, merge quality, and coordinator throughput at that scale are untested.

- **Single coordinator.** One server handles all merges. It's stateless (all data in S3) and can be restarted, but it's a single point of failure. Peer-to-peer topology is a natural extension.

- **No security.** No authentication on delta pushes. Anyone who finds the coordinator can submit garbage. Real deployment needs signed contributions and verification.

- **Coordinator persistence is fragile.** MinIO data on a Docker volume works but isn't robust. A cloud-native deployment with proper S3 (R2, AWS) would be more reliable.

## Future directions

- **Scale to 7B with MoE.** Expert sharding across nodes — each node holds one expert + shared layers. Most parameters never cross the network. DiPaCo-style document routing.
- **Delta streaming.** Push every 5-10 steps instead of 50. Smaller deltas, less staleness, fresher gradients. Requires fast coordinator merge (<1s).
- **GPU aggregation.** Move weighted averaging to GPU (burn tensors). Reduce merge time from 30s to <1s, enabling higher-frequency checkpoints.
- **Shard rotation.** Background rotation of training data shards to increase diversity without blocking the GPU. Currently shards are fixed per node.
- **Desktop app.** The Tauri shell exists. A one-click "contribute to training" app for non-technical users.
- **Browser training.** WebAssembly node exists. Train in a browser tab. Currently limited by WebGPU maturity.

## Contributing

We're looking for people to help with:

- **Run experiments on consumer GPUs.** We've tested on A40s and Macs. What happens with RTX 3060s, 4070s, mixed AMD/NVIDIA clusters? How does residential internet affect quality?
- **Scale testing.** 10, 50, 100 nodes. Does the protocol hold? Where does the coordinator bottleneck?
- **Bigger models.** 1B and 7B training runs. Do the same optimizations (constant lr, batch=4, high retention) work at scale?
- **Compression research.** Better ways to shrink deltas without losing signal. Structured sparsity, learned compression, hybrid approaches.
- **Infrastructure hardening.** Better persistence, fault tolerance, authentication.

If you're interested, open an issue or reach out. The codebase is ~15K lines of Rust, well-structured, with Docker images that just work.

## Related work

- [DiLoCo](https://arxiv.org/abs/2311.08105) (Google, 2023) — inner-outer optimization, requires synchronous outer steps
- [INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) (Prime Intellect) — 10B across continents, synchronous
- [Psyche](https://nousresearch.com/nous-psyche/) (Nous Research) — decentralized training, requires synchronization
- [Hivemind](https://github.com/learning-at-home/hivemind) (Together.ai) — PyTorch library for decentralized training

Distrain's contribution: fully asynchronous merge with zero synchronization, pure Rust single-binary implementation, auto-tuning for heterogeneous hardware, and systematic measurement of the distributed-vs-single gap.

## License

MIT
