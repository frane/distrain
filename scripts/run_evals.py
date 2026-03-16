#!/usr/bin/env python3
"""Evaluate checkpoints on standard benchmarks.

Usage:
    # Single checkpoint
    python scripts/run_evals.py --checkpoint checkpoints/v500/model.safetensors \
                                --tasks mmlu,hellaswag,arc_challenge

    # Track over training (eval curves)
    python scripts/run_evals.py --checkpoint-dir checkpoints/ \
                                --versions 100,200,300,400,500 \
                                --tasks mmlu,hellaswag

    # Quick sanity check (limited examples)
    python scripts/run_evals.py --checkpoint latest --tasks hellaswag --limit 100

    # Random baseline (untrained model)
    python scripts/run_evals.py --checkpoint random --model-size 125m --tasks hellaswag
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add python/ to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from distrain.eval.benchmarks import (
    FULL_TASKS,
    QUICK_TASKS,
    CheckpointEvalResult,
    evaluate_checkpoint,
    load_eval_results,
    save_eval_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def find_checkpoint(checkpoint_dir: Path, version: int) -> str:
    """Find a checkpoint file for a given version."""
    candidates = [
        checkpoint_dir / f"v{version}" / "model.safetensors",
        checkpoint_dir / f"step_{version}" / "model.safetensors",
        checkpoint_dir / f"v{version}.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"No checkpoint found for version {version} in {checkpoint_dir}")


def plot_eval_curves(results: list[CheckpointEvalResult], output_path: Path) -> None:
    """Plot eval scores vs checkpoint version."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    # Group by task
    task_scores: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        if r.checkpoint_version is None:
            continue
        for er in r.results:
            task_scores.setdefault(er.task, []).append((r.checkpoint_version, er.score))

    if not task_scores:
        logger.warning("No versioned results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for task, points in sorted(task_scores.items()):
        points.sort(key=lambda x: x[0])
        versions = [p[0] for p in points]
        scores = [p[1] for p in points]
        ax.plot(versions, scores, marker="o", label=task)

    ax.set_xlabel("Checkpoint Version")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Scores Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved eval curve plot to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on standard benchmarks")

    # Checkpoint selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to checkpoint, or 'random' for untrained baseline")
    group.add_argument("--checkpoint-dir", type=str, help="Directory containing versioned checkpoints")

    # Tasks
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: full suite)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick eval only (hellaswag)")

    # Model
    parser.add_argument("--model-size", type=str, default="125m",
                        choices=["125m", "1b", "7b", "13b"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)

    # Multi-version
    parser.add_argument("--versions", type=str, default=None,
                        help="Comma-separated checkpoint versions to evaluate")

    # Options
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for quick testing)")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory to save results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate eval curve plot (requires matplotlib)")

    args = parser.parse_args()

    # Resolve tasks
    if args.quick:
        tasks = QUICK_TASKS
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = FULL_TASKS

    output_dir = Path(args.output_dir)
    all_results: list[CheckpointEvalResult] = []

    if args.checkpoint:
        # Single checkpoint evaluation
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            model_size=args.model_size,
            tasks=tasks,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        all_results.append(result)
        print("\n" + result.summary_table())
        save_eval_results(result, output_dir)

    elif args.checkpoint_dir:
        # Multi-version evaluation
        checkpoint_dir = Path(args.checkpoint_dir)
        if not args.versions:
            logger.error("--versions required when using --checkpoint-dir")
            sys.exit(1)

        versions = [int(v.strip()) for v in args.versions.split(",")]

        for version in versions:
            try:
                ckpt_path = find_checkpoint(checkpoint_dir, version)
            except FileNotFoundError as e:
                logger.warning(str(e))
                continue

            logger.info(f"Evaluating version {version}: {ckpt_path}")
            result = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                model_size=args.model_size,
                tasks=tasks,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
                checkpoint_version=version,
            )
            all_results.append(result)
            print("\n" + result.summary_table())
            save_eval_results(result, output_dir)

    # Plot eval curves if requested
    if args.plot and len(all_results) > 1:
        plot_eval_curves(all_results, output_dir / "eval_curves.png")

    # Also load any previously saved results for the plot
    if args.plot and len(all_results) <= 1:
        existing = list(output_dir.glob("v*.json"))
        if existing:
            loaded = [load_eval_results(p) for p in existing]
            loaded.extend(all_results)
            plot_eval_curves(loaded, output_dir / "eval_curves.png")


if __name__ == "__main__":
    main()
