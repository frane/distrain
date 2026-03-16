#!/usr/bin/env python3
"""Direct Preference Optimization training.

Usage:
    # LoRA DPO (recommended)
    python scripts/run_dpo.py \
        --sft-model models/distrain-7b-sft/final/model.safetensors \
        --datasets UltraFeedback-binarized,Orca-DPO \
        --output models/distrain-7b-chat-lora \
        --beta 0.1 --lr 5e-7 --use-lora

    # Smoke test on synthetic data
    python scripts/run_dpo.py --synthetic --model-size 125m --output models/test-dpo
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from distrain.alignment.dpo import DPOConfig, DPOTrainer
from distrain.alignment.preference_data import (
    PreferenceDataset,
    PreferencePair,
)
from distrain.model.config import CONFIGS
from distrain.model.transformer import DistrainTransformer
from distrain.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def create_synthetic_preferences(num_examples: int = 50, max_seq_len: int = 64) -> PreferenceDataset:
    """Create synthetic preference data for testing."""
    pairs = []
    for i in range(num_examples):
        prompt = list(range(1, 17))  # 16 prompt tokens
        chosen = prompt + list(range(17, 33))  # 16 chosen response tokens
        rejected = prompt + list(range(33, 49))  # 16 different rejected tokens
        pairs.append(PreferencePair(
            prompt_ids=prompt,
            chosen_ids=chosen,
            rejected_ids=rejected,
        ))
    return PreferenceDataset(pairs, max_seq_len=max_seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training")

    parser.add_argument("--sft-model", type=str, default=None, help="Path to SFT model checkpoint")
    parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated HF dataset names")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)

    parser.add_argument("--output", type=str, default="models/dpo")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    config = CONFIGS[args.model_size]

    # Load SFT model as policy
    policy = DistrainTransformer(config)
    if args.sft_model:
        state_dict = load_checkpoint(args.sft_model, device=args.device)
        policy.load_state_dict(state_dict)
        logger.info(f"Loaded SFT model from {args.sft_model}")

    # Reference model is a frozen copy
    reference = DistrainTransformer(config)
    reference.load_state_dict(policy.state_dict())

    # Load data
    if args.synthetic:
        dataset = create_synthetic_preferences(max_seq_len=args.max_seq_len)
    elif args.datasets:
        from distrain.alignment.preference_data import parse_preference_pairs
        from distrain.alignment.data import load_hf_dataset
        from distrain.data.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        all_pairs = []
        for name in args.datasets.split(","):
            raw = load_hf_dataset(name.strip(), max_examples=args.max_examples)
            pairs = parse_preference_pairs(raw, tokenizer, max_seq_len=args.max_seq_len)
            all_pairs.extend(pairs)
        dataset = PreferenceDataset(all_pairs, max_seq_len=args.max_seq_len)
    else:
        logger.error("Provide --datasets or --synthetic")
        sys.exit(1)

    logger.info(f"Dataset: {len(dataset)} preference pairs")

    dpo_config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        save_every_n_steps=args.save_every,
        log_every_n_steps=args.log_every,
        output_dir=args.output,
    )

    trainer = DPOTrainer(policy, reference, dataset, dpo_config, device=args.device)
    metrics = trainer.train()

    output_dir = Path(args.output)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
