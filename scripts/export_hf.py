#!/usr/bin/env python3
"""Export DistrainTransformer to HuggingFace Llama-compatible format.

Usage:
    python scripts/export_hf.py \
        --checkpoint models/distrain-7b-chat/final/model.safetensors \
        --output hf_export/distrain-7b-chat/ \
        --model-size 7b

    # Produces:
    # hf_export/distrain-7b-chat/
    # |-- config.json
    # |-- model.safetensors
    # |-- tokenizer_config.json
    # |-- generation_config.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to model.safetensors")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model-size", default="125m", choices=["125m", "1b", "7b", "13b"])
    args = parser.parse_args()

    from distrain.export.weight_mapping import export_to_hf
    from distrain.model.config import CONFIGS
    from distrain.training.checkpointing import load_checkpoint

    config = CONFIGS[args.model_size]
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)
    logger.info(f"Loaded {len(state_dict)} tensors")

    output_dir = Path(args.output)
    export_to_hf(state_dict, config, output_dir)
    logger.info(f"Export complete: {output_dir}")


if __name__ == "__main__":
    main()
