# Distrain

Asynchronous distributed LLM training with zero synchronization.

Multiple GPUs train the same model without coordination. Each device trains locally, pushes weight updates to a coordinator, and the coordinator merges them. No AllReduce, no synchronous rounds, no waiting for the slowest node. A datacenter GPU and a gaming PC participate in the same training run — each auto-configures and contributes proportionally.

Written entirely in Rust. Single binary, no Python runtime, no PyTorch. Runs on CUDA, Metal, Vulkan, and CPU.

**Long-term vision:** volunteer computing for AI training, like SETI@home. We're not there yet — current bandwidth and memory requirements limit practical participation to devices with dedicated GPUs and decent network connections. But the protocol is designed for that future.

## Results

We systematically measured the overhead of distributed vs single-GPU training on a 125M parameter transformer across seven experiment curves:

| Curve | Hardware | Plateau Loss | Gap to BL | Key Change |
|-------|----------|-------------|-----------|------------|
| Baseline (A) | 1× RTX 4000 Ada | 4.8 | — | Single GPU, batch=4, lr=3e-4 |
| B | 3× Mac (2 GPU + 1 CPU) | 6.7 | +1.9 | First distributed, heterogeneous consumer hardware |
| C | 3× mixed RTX | 6.4 | +1.6 | Outer LR→1.0, compression tuning |
| D | 3× RTX A4000 | 6.0 | +1.2 | Patience triggers, all nodes contributing |
| E | 3× A40 | 5.8 | +1.0 | Continuous training (GPU never idles) |
| H | 3× A40 | 5.98 | +1.2 | Clean comparison: same hyperparams as baseline |
| G | 3× A40 | 6.4 | +1.6 | Loss-based lr (decays as model converges) |

Validation loss on held-out data matches training loss within 0.4% across all curves, confirming no overfitting at <2% data utilization.

**Key findings:**

- **The gap is 1.0-1.2 points** between distributed (3 nodes) and single-GPU baseline. This comes from compression loss (~10% signal per round) and checkpoint merge staleness (~30s of stale training per checkpoint).

- **More contributors = better quality.** 3-contribution checkpoints consistently outperform 2-contribution ones. The weighted average of diverse gradients from multiple nodes improves merge quality.

- **Constant lr matches baseline best.** Loss-based lr scheduling plateaus 0.4 points higher than constant lr. The decay reduces learning rate before the model has fully converged.

- **Consumer hardware works.** Curve B trained across a 10× throughput gap (Apple Silicon GPUs + Intel CPU) for 2,928 checkpoints. The protocol handles heterogeneous devices transparently.

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

## Current limitations

- **Delta size.** Raw deltas for 125M model: ~400MB. For 7B: ~14GB. Heavy compression reduces this but loses signal. Low-rank decomposition (planned) could achieve 20-200x compression with minimal signal loss.
- **Memory.** Training 7B requires 20-40GB VRAM minimum. Most consumer devices can't participate at meaningful model sizes.
- **Scale.** Tested with 3 GPUs. The protocol is designed for 50-1000+ nodes but hasn't been validated at that scale.
- **Single coordinator.** Current architecture has one coordinator. Peer-to-peer topology is a natural extension.

## Model presets

Training a different model size is just an environment variable:

| Preset | Params | Hidden | Layers | Heads | KV Heads | FFN | VRAM needed |
|--------|--------|--------|--------|-------|----------|-----|-------------|
| `micro-test` | 64K | 64 | 2 | 2 | 2 | 256 | <1 GB |
| `tiny` | 125M | 768 | 12 | 12 | 4 | 2048 | ~8 GB |
| `small` | 1.13B | 2048 | 24 | 16 | 4 | 5504 | ~40 GB |
| `medium` | 7B | 4096 | 32 | 32 | 8 | 11008 | ~80 GB |
| `large` | 13B | 5120 | 40 | 40 | 8 | 13824 | ~160 GB |

Set `PRESET=small` on the coordinator to bootstrap a 1B model. Nodes auto-detect the architecture from the checkpoint — no configuration needed on their side.

All presets use the Mistral v0.3 tokenizer (32,768 vocab).

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

This downloads FineWeb-Edu from HuggingFace (~10B tokens), tokenizes it with batch encoding (~1 hour), and uploads 1,102 shards to MinIO. The data is stored on the persistent volume — you only need to do this once.

Available datasets:
- `fineweb-edu-10bt` — 10B tokens, good for 125M-1B models
- `fineweb-edu-100bt` — 100B tokens, for 7B+ models
- `test` — random test shards (no download, instant)

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

Each node auto-detects its GPU, computes optimal batch size from VRAM and model architecture, scales learning rate accordingly, and starts training immediately. No manual configuration. Start as many nodes as you want — the protocol handles heterogeneous hardware automatically.

### Local development

```bash
cargo build --release -p distrain-coordinator -p distrain-node

# Start MinIO (local S3)
docker compose -f docker/docker-compose.yml up -d minio

# Prepare training data
pip install datasets tokenizers numpy tqdm
python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb --upload

# Bootstrap a model (change --preset for different sizes)
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

## Related work

- [DiLoCo](https://arxiv.org/abs/2311.08105) (Google, 2023) — inner-outer optimization, requires synchronous outer steps
- [INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) (Prime Intellect) — 10B across continents, synchronous
- [Psyche](https://nousresearch.com/nous-psyche/) (Nous Research) — decentralized training, requires synchronization
- [Hivemind](https://github.com/learning-at-home/hivemind) (Together.ai) — PyTorch library for decentralized training

Distrain's contribution: fully asynchronous merge with zero synchronization, auto-tuning for heterogeneous hardware, and systematic measurement of the distributed-vs-single gap.

## License

MIT
