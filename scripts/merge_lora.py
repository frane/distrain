#!/usr/bin/env python3
"""Merge LoRA adapters back into base model weights.

Usage:
    python scripts/merge_lora.py \
        --base models/distrain-7b-sft/final/model.safetensors \
        --lora models/distrain-7b-chat-lora/final/model.safetensors \
        --model-size 7b \
        --output models/distrain-7b-chat/model.safetensors
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from distrain.alignment.sft import SFTConfig, apply_lora, merge_lora
from distrain.model.config import CONFIGS
from distrain.model.transformer import DistrainTransformer
from distrain.training.checkpointing import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base", required=True, help="Base model checkpoint")
    parser.add_argument("--lora", required=True, help="LoRA model checkpoint (saved with LoRA weights)")
    parser.add_argument("--model-size", default="125m", choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    args = parser.parse_args()

    config = CONFIGS[args.model_size]

    # Load base model
    logger.info(f"Loading base model from {args.base}")
    model = DistrainTransformer(config)
    base_state = load_checkpoint(args.base)
    model.load_state_dict(base_state)

    # Apply LoRA structure
    sft_config = SFTConfig(use_lora=True, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    model = apply_lora(model, sft_config)

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from {args.lora}")
    lora_state = load_checkpoint(args.lora)
    model.load_state_dict(lora_state)

    # Merge LoRA back into base
    logger.info("Merging LoRA adapters...")
    model = merge_lora(model)

    # Save merged model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.state_dict(), output_path)
    logger.info(f"Saved merged model to {output_path}")


if __name__ == "__main__":
    main()
