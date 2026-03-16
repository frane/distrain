#!/usr/bin/env python3
"""Benchmark gradient compression ratios and speed.

Measures compression of realistic outer gradients at various zstd levels
and with different quantization strategies.

Usage:
    python scripts/benchmark_compression.py                  # 125M model
    python scripts/benchmark_compression.py --model-size 1b  # 1B model
"""

import argparse
import io
import logging
import time

import torch
import zstandard as zstd

from distrain.model.config import CONFIGS
from distrain.model.transformer import DistrainTransformer
from distrain.training.diloco import DiLoCoInnerTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_realistic_gradient(model_size: str, steps: int = 10) -> dict[str, torch.Tensor]:
    """Train a few steps to get a realistic outer gradient."""
    config = CONFIGS[model_size]
    model = DistrainTransformer(config)
    tc = TrainingConfig(
        inner_steps=steps, batch_size=2, gradient_accumulation_steps=1,
        sequence_length=64, learning_rate=1e-3, warmup_steps=2,
        total_steps=100, use_amp=False,
    )
    trainer = DiLoCoInnerTrainer(model, config, tc, torch.device("cpu"))
    trainer.start_round()
    for i in range(steps):
        batch = torch.randint(0, config.vocab_size, (tc.batch_size, tc.sequence_length))
        trainer.train_step(batch, i)
    return trainer.compute_outer_gradient()


def measure_compression(
    grad: dict[str, torch.Tensor],
    dtype: torch.dtype | None,
    zstd_level: int,
) -> dict:
    """Compress a gradient and measure size/speed."""
    if dtype is not None:
        grad_q = {k: v.to(dtype) for k, v in grad.items()}
    else:
        grad_q = grad

    # Serialize
    buf = io.BytesIO()
    torch.save(grad_q, buf)
    raw_bytes = buf.getvalue()
    raw_size = len(raw_bytes)

    # Compress
    t0 = time.perf_counter()
    compressor = zstd.ZstdCompressor(level=zstd_level)
    compressed = compressor.compress(raw_bytes)
    compress_time = time.perf_counter() - t0
    compressed_size = len(compressed)

    # Decompress
    t0 = time.perf_counter()
    decompressor = zstd.ZstdDecompressor()
    decompressed = decompressor.decompress(compressed)
    decompress_time = time.perf_counter() - t0

    # Verify roundtrip
    grad_back = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=True)
    for k in grad_q:
        assert torch.equal(grad_q[k], grad_back[k]), f"Roundtrip mismatch in {k}"

    # Original FP32 size for ratio
    fp32_size = sum(v.numel() * 4 for v in grad.values())

    return {
        "raw_serialized_mb": raw_size / 1e6,
        "compressed_mb": compressed_size / 1e6,
        "fp32_size_mb": fp32_size / 1e6,
        "ratio_vs_fp32": fp32_size / compressed_size,
        "ratio_vs_raw": raw_size / compressed_size,
        "compress_sec": compress_time,
        "decompress_sec": decompress_time,
    }


def measure_quality(
    grad_fp32: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict:
    """Measure quality loss from quantization."""
    grad_q = {k: v.to(dtype).to(torch.float32) for k, v in grad_fp32.items()}
    errors = []
    for k in grad_fp32:
        err = (grad_fp32[k] - grad_q[k]).abs()
        errors.append({
            "param": k,
            "max_abs_error": err.max().item(),
            "mean_abs_error": err.mean().item(),
            "relative_error": (err / (grad_fp32[k].abs() + 1e-8)).mean().item(),
        })
    total_norm_orig = sum(v.norm().item() ** 2 for v in grad_fp32.values()) ** 0.5
    total_norm_q = sum(v.norm().item() ** 2 for v in grad_q.values()) ** 0.5
    cos_sim = sum(
        (grad_fp32[k] * grad_q[k]).sum().item() for k in grad_fp32
    ) / (total_norm_orig * total_norm_q + 1e-8)

    return {
        "cosine_similarity": cos_sim,
        "norm_ratio": total_norm_q / (total_norm_orig + 1e-8),
        "max_relative_error": max(e["relative_error"] for e in errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="125m", choices=list(CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=10, help="Training steps for gradient")
    args = parser.parse_args()

    config = CONFIGS[args.model_size]
    fp32_model_size = config.param_count() * 4 / 1e6
    logger.info(f"Model: {args.model_size} ({config.param_count_str()}, "
                f"FP32 size: {fp32_model_size:.1f} MB)")
    logger.info(f"Generating realistic gradient ({args.steps} steps)...\n")

    grad = generate_realistic_gradient(args.model_size, args.steps)

    # Compression benchmarks
    pipelines = [
        ("FP32 + zstd-1", None, 1),
        ("FP32 + zstd-3", None, 3),
        ("FP32 + zstd-5", None, 5),
        ("FP32 + zstd-9", None, 9),
        ("BF16 + zstd-1", torch.bfloat16, 1),
        ("BF16 + zstd-3", torch.bfloat16, 3),
        ("BF16 + zstd-5", torch.bfloat16, 5),
        ("BF16 + zstd-9", torch.bfloat16, 9),
    ]

    print(f"{'Pipeline':<20} {'Compressed':>12} {'Ratio vs FP32':>14} "
          f"{'Compress':>10} {'Decompress':>12}")
    print("-" * 72)

    for name, dtype, level in pipelines:
        r = measure_compression(grad, dtype, level)
        print(f"{name:<20} {r['compressed_mb']:>9.1f} MB {r['ratio_vs_fp32']:>12.1f}x "
              f"{r['compress_sec']:>9.3f}s {r['decompress_sec']:>11.3f}s")

    # Quality benchmarks
    print(f"\n{'Quantization Quality':}")
    print("-" * 60)
    print(f"{'Dtype':<15} {'Cosine Sim':>12} {'Norm Ratio':>12} {'Max Rel Err':>14}")
    print("-" * 60)

    for dtype_name, dtype in [("BF16", torch.bfloat16), ("FP16", torch.float16)]:
        q = measure_quality(grad, dtype)
        print(f"{dtype_name:<15} {q['cosine_similarity']:>12.6f} "
              f"{q['norm_ratio']:>12.6f} {q['max_relative_error']:>14.6f}")

    # Extrapolations
    print(f"\nExtrapolated sizes for 7B model ({CONFIGS['7b'].param_count_str()}):")
    scale = CONFIGS["7b"].param_count() / config.param_count()
    for name, dtype, level in [("BF16 + zstd-3", torch.bfloat16, 3),
                                ("FP32 + zstd-3", None, 3)]:
        r = measure_compression(grad, dtype, level)
        est = r["compressed_mb"] * scale
        print(f"  {name}: ~{est / 1000:.1f} GB")

    print("\nRecommendation: BF16 + zstd-3 (best ratio/speed tradeoff)")


if __name__ == "__main__":
    main()
