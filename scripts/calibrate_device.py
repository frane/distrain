#!/usr/bin/env python3
"""Quick device benchmark. Shows what your hardware can do.

Usage:
    python scripts/calibrate_device.py                     # auto-detect device, 125M model
    python scripts/calibrate_device.py --model-size 1b     # test with 1B model
    python scripts/calibrate_device.py --device cpu         # force CPU
"""

import argparse
import logging
import sys

import torch

from distrain.model.config import CONFIGS
from distrain.model.transformer import DistrainTransformer
from distrain.training.calibration import calibrate, detect_best_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate device for Distrain training")
    parser.add_argument("--model-size", default="125m", choices=list(CONFIGS.keys()),
                        help="Model size preset (default: 125m)")
    parser.add_argument("--device", default=None, help="Force device (cpu, cuda:0, mps)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--target-interval", type=float, default=60.0,
                        help="Target push interval in seconds (default: 60)")
    args = parser.parse_args()

    config = CONFIGS[args.model_size]
    logger.info(f"Model: {args.model_size} ({config.param_count_str()} params)")

    if args.device:
        device = torch.device(args.device)
        dtype = "float32" if args.device == "cpu" else None
    else:
        device = None
        dtype = None

    # Check if model fits
    param_bytes = config.param_count() * 4  # FP32
    logger.info(f"Model memory (FP32): {param_bytes / 1e9:.1f} GB")

    logger.info("Creating model...")
    model = DistrainTransformer(config)

    logger.info("Running calibration...")
    profile = calibrate(
        model,
        batch_size=args.batch_size,
        seq_len=min(args.seq_len, config.max_seq_len),
        target_push_interval_secs=args.target_interval,
        device=device,
        dtype=dtype,
    )

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Device:        {profile.device_name} ({profile.device})")
    print(f"Precision:     {profile.dtype}")
    if profile.vram_gb:
        print(f"VRAM:          {profile.vram_gb:.1f} GB")
    print(f"RAM:           {profile.ram_gb:.1f} GB")
    print(f"Speed:         {profile.tokens_per_sec:,.0f} tokens/sec "
          f"({profile.secs_per_step:.3f} sec/step)")
    print(f"H_mini:        {profile.h_mini} steps per push")
    print(f"Push interval: ~{profile.estimated_push_interval_secs:.0f}s")
    print(f"Weight:        {profile.estimated_weight:.2f}x per push")
    print()

    # Context: how this device compares in a network
    h100_tok_sec = 18000  # estimated H100 throughput on 7B
    relative = profile.tokens_per_sec / h100_tok_sec
    print(f"Relative to H100: {relative:.1%} throughput")
    print(f"In a network of 30 H100s, your device adds ~{relative / 30 * 100:.1f}% extra signal")
    print("=" * 60)


if __name__ == "__main__":
    main()
