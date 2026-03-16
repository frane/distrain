#!/usr/bin/env python3
"""Supervised fine-tuning on chat datasets.

Usage:
    # Full fine-tune on local data
    python scripts/run_sft.py \
        --base-model checkpoints/v_final/model.safetensors \
        --datasets OpenHermes-2.5,SlimOrca \
        --output models/distrain-7b-sft \
        --epochs 3 --lr 2e-5

    # LoRA (faster, lower VRAM)
    python scripts/run_sft.py \
        --base-model checkpoints/v_final/model.safetensors \
        --datasets OpenHermes-2.5 \
        --output models/distrain-7b-sft-lora \
        --use-lora --lora-rank 64 --epochs 3

    # Smoke test on synthetic data (no download)
    python scripts/run_sft.py --synthetic --model-size 125m --output models/test-sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from distrain.alignment.chat_template import Message
from distrain.alignment.data import (
    ChatDataset,
    ChatExample,
    tokenize_conversations,
)
from distrain.alignment.sft import SFTConfig, SFTTrainer
from distrain.data.tokenizer import Tokenizer
from distrain.model.config import CONFIGS
from distrain.model.transformer import DistrainTransformer
from distrain.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def create_synthetic_dataset(tokenizer, num_examples: int = 100, max_seq_len: int = 256) -> ChatDataset:
    """Create a synthetic chat dataset for testing."""
    conversations = []
    responses = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a versatile programming language.",
        "The Earth orbits the Sun at a distance of about 93 million miles.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    ]
    questions = [
        "What is machine learning?",
        "Tell me about Python.",
        "How far is the Earth from the Sun?",
        "At what temperature does water boil?",
        "What is photosynthesis?",
    ]

    for i in range(num_examples):
        idx = i % len(questions)
        conv = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=questions[idx]),
            Message(role="assistant", content=responses[idx]),
        ]
        conversations.append(conv)

    return ChatDataset(
        tokenize_conversations(conversations, tokenizer, max_seq_len=max_seq_len),
        max_seq_len=max_seq_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning")

    # Model
    parser.add_argument("--base-model", type=str, default=None, help="Path to base model checkpoint")
    parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--device", type=str, default="cpu")

    # Data
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated HF dataset names")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples per dataset")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--template", type=str, default="chatml", choices=["chatml", "llama", "zephyr"])

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)

    # LoRA
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)

    # Output
    parser.add_argument("--output", type=str, default="models/sft")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    # Load model
    config = CONFIGS[args.model_size]
    model = DistrainTransformer(config)
    if args.base_model:
        state_dict = load_checkpoint(args.base_model, device=args.device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded base model from {args.base_model}")

    tokenizer = Tokenizer()

    # Load data
    if args.synthetic:
        logger.info("Using synthetic chat dataset")
        dataset = create_synthetic_dataset(tokenizer, max_seq_len=args.max_seq_len)
    elif args.datasets:
        from distrain.alignment.data import load_chat_dataset
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        dataset = load_chat_dataset(
            dataset_names, tokenizer,
            template=args.template,
            max_seq_len=args.max_seq_len,
            max_examples_per_dataset=args.max_examples,
        )
    else:
        logger.error("Provide --datasets or --synthetic")
        sys.exit(1)

    logger.info(f"Dataset: {len(dataset)} examples, max_seq_len={args.max_seq_len}")

    # Configure trainer
    sft_config = SFTConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        save_every_n_steps=args.save_every,
        log_every_n_steps=args.log_every,
        output_dir=args.output,
    )

    trainer = SFTTrainer(model, dataset, sft_config, device=args.device)
    metrics = trainer.train()

    # Save metrics
    output_dir = Path(args.output)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
