#!/usr/bin/env python3
"""Interactive CLI chat with a Distrain model.

Usage:
    python scripts/chat.py --model models/distrain-7b-chat/final/model.safetensors
    python scripts/chat.py --model random --model-size 125m  # test with untrained model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a Distrain model")
    parser.add_argument("--model", required=True, help="Path to model, or 'random' for untrained")
    parser.add_argument("--model-size", default="125m", choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    args = parser.parse_args()

    import torch
    from distrain.alignment.chat_template import Message, format_chatml
    from distrain.data.tokenizer import Tokenizer
    from distrain.inference.generate import GenerationConfig, generate
    from distrain.model.config import CONFIGS
    from distrain.model.transformer import DistrainTransformer
    from distrain.training.checkpointing import load_checkpoint

    config = CONFIGS[args.model_size]
    model = DistrainTransformer(config)

    if args.model != "random":
        model_path = Path(args.model)
        if model_path.is_dir():
            model_path = model_path / "model.safetensors"
        state_dict = load_checkpoint(str(model_path), device=args.device)
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()
    tokenizer = Tokenizer()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        stop_sequences=["<|im_end|>"],
    )

    print(f"Distrain Chat ({args.model_size}, {config.param_count_str()})")
    print(f"Device: {args.device} | Temperature: {args.temperature}")
    print("Type 'quit' or Ctrl+C to exit.\n")

    conversation: list[Message] = [Message(role="system", content=args.system)]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            conversation = [Message(role="system", content=args.system)]
            print("[Conversation cleared]\n")
            continue

        conversation.append(Message(role="user", content=user_input))
        prompt = format_chatml(conversation, add_generation_prompt=True)

        response = generate(model, tokenizer, prompt, gen_config)
        response = response.strip()

        print(f"Distrain: {response}\n")
        conversation.append(Message(role="assistant", content=response))


if __name__ == "__main__":
    main()
