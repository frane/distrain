# Distrain

Distributed LLM training on consumer hardware. Like SETI@home but for training language models.

The idea is simple: anyone with a computer — GPU, CPU, laptop, whatever — can contribute to training an LLM from scratch. Your device trains locally, compresses the weight update, and pushes it to a coordinator that merges everything into a shared checkpoint. No synchronization, no waiting. A fast GPU pushes every 30 seconds, a slow laptop pushes every 10 minutes. Both contribute.

The whole thing is written in Rust. Single binary, no Python runtime, no PyTorch. Compiles to native (CUDA, Metal, Vulkan, CPU) and WebAssembly.

## Does it actually work?

Yes. We trained a 125M parameter model to convergence on 3 MacBooks over a LAN:

- **M2 Pro 32GB** — GPU via Metal
- **MacBook Air M2 24GB** — GPU via Metal
- **Intel i7 MacBook Pro 2020 32GB** — CPU only (no usable GPU)

Loss went from **674 (random init) → 6.3 (converged)** over 2,928 checkpoint versions. The model generates semi-coherent English text.

We're currently training a 1B model on the same setup. Loss is at **~65** (down from 1,669 at init) after 48 checkpoints. Each push compresses a 4.5 GB model delta down to ~25 MB — that's 175x compression.

It's slow with 3 devices (about 26 tokens/second combined for 1B). That's the point — it's designed for hundreds or thousands of nodes, not three.

<!-- TODO: loss curve screenshot -->

## The protocol

There's no AllReduce, no parameter server, no synchronous rounds. The merge operation is a CRDT (commutative, associative, idempotent), so deltas can arrive in any order and the result is the same.

Each node:
1. Downloads the latest checkpoint
2. Trains for a while (auto-calibrated per device)
3. Compresses the delta: top-k sparsification (keep top 0.5-7% by magnitude) → error feedback (nothing permanently lost) → INT8 quantization → zstd
4. Uploads to object storage, pings the coordinator
5. Coordinator merges deltas with staleness-weighted averaging + Nesterov momentum
6. New checkpoint appears, repeat

Stale deltas (computed against old checkpoints) get exponentially less weight: `0.9^staleness`. A delta that's 10 versions behind is discarded entirely. This means slow devices naturally contribute less but never hold anything back.

## Running it

You need Rust, a MinIO instance (or any S3-compatible storage), and some tokenized training data.

```bash
# Build
cargo build --release -p distrain-coordinator -p distrain-node

# Start MinIO
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

To add more nodes, point them at the coordinator IP in `node.toml` and run the same binary. Nodes automatically get assigned different training data via deterministic hashing — no coordinator involvement needed.

## What's here

```
coordinator/        HTTP server (Axum) + aggregation logic
core/model/         Transformer implementation, compression, checkpointing
core/shared/        Storage client, config, shared types
node/cli/           The training node (this is the main thing)
node/desktop/       Tauri desktop app (experimental)
node/browser/       WebAssembly version (experimental)
node/ffi/           C FFI for mobile (experimental)
scripts/            Data prep, eval, benchmarks
docker/             Local dev stack (MinIO, Prometheus, Grafana)
```

The CLI node and coordinator are production-quality. Desktop, browser, and mobile are scaffolded but not battle-tested.

## Model

Standard decoder-only transformer (GQA + RoPE + SwiGLU + RMSNorm). Implemented in Rust via [Burn](https://burn.dev). Presets from 1M to 13B parameters.

The node auto-detects GPU vs CPU, picks the faster backend, calibrates batch size (retries on OOM), and figures out how many steps to run per push.

## Related work

This isn't the first attempt at decentralized training:

- [DiLoCo](https://arxiv.org/abs/2311.08105) (Google, 2023) — the inner-outer optimization idea that inspired this
- [INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) (Prime Intellect) — trained a 10B model across continents, but with datacenter GPUs and synchronous outer steps
- [Psyche](https://nousresearch.com/nous-psyche/) (Nous Research) — raised $50M, trained 36B, supports consumer GPUs but still synchronizes
- [Hivemind](https://github.com/learning-at-home/hivemind) (Together.ai) — PyTorch decentralized training library, mostly used for inference

What's different here: fully asynchronous CRDT merge (zero synchronization), pure Rust single binary (zero dependencies), and it runs on anything including CPUs and browsers — not just datacenter GPUs with the occasional 4090.

## License

MIT
