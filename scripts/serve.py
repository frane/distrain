#!/usr/bin/env python3
"""Start OpenAI-compatible API server for Distrain models.

Usage:
    python scripts/serve.py --model models/distrain-7b-chat/final/model.safetensors
    python scripts/serve.py --model models/distrain-7b-chat/ --port 8080 --device cuda
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
    parser = argparse.ArgumentParser(description="Distrain inference server")
    parser.add_argument("--model", required=True, help="Path to model checkpoint or directory")
    parser.add_argument("--model-size", default="125m", choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="distrain-7b")
    args = parser.parse_args()

    import torch
    from distrain.data.tokenizer import Tokenizer
    from distrain.inference.server import ServerConfig, create_app
    from distrain.model.config import CONFIGS
    from distrain.model.transformer import DistrainTransformer
    from distrain.training.checkpointing import load_checkpoint

    # Resolve checkpoint path
    model_path = Path(args.model)
    if model_path.is_dir():
        model_path = model_path / "model.safetensors"

    config = CONFIGS[args.model_size]
    model = DistrainTransformer(config)
    state_dict = load_checkpoint(str(model_path), device=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    logger.info(f"Loaded model: {args.model_size} ({config.param_count_str()}) on {args.device}")

    tokenizer = Tokenizer()

    server_config = ServerConfig(host=args.host, port=args.port, model_name=args.model_name)
    app = create_app(model, tokenizer, server_config)

    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
