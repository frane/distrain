#!/usr/bin/env python3
"""Simulate CRDT continuous merge and compare against baseline + DiLoCo.

This is the make-or-break experiment. Proves the CRDT protocol converges.

Usage:
    python scripts/simulate_continuous.py                          # defaults
    python scripts/simulate_continuous.py --nodes 4 --total-steps 2000
    python scripts/simulate_continuous.py --model-size 125m --total-steps 8000
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from distrain.model.config import CONFIGS, ModelConfig
from distrain.model.transformer import DistrainTransformer
from distrain.training.crdt import GradientAccumulator
from distrain.training.diloco import NesterovOuterOptimizer, cosine_lr_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Use a small model for fast simulation
TINY = ModelConfig(
    hidden_dim=128, num_layers=4, num_heads=4, num_kv_heads=2,
    ffn_hidden_dim=256, vocab_size=512, max_seq_len=128,
)


@dataclass
class SimConfig:
    model_config: ModelConfig = field(default_factory=lambda: TINY)
    num_nodes: int = 8
    total_steps: int = 4000
    h_diloco: int = 500
    h_mini: int = 50
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    inner_lr: float = 1e-3
    min_inner_lr: float = 1e-4
    warmup_steps: int = 100
    batch_size: int = 4
    seq_len: int = 64
    min_contributions: int = 4
    staleness_decay: float = 0.9
    node_speeds: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4])
    device: str = "cpu"


@dataclass
class StepRecord:
    step: int
    loss: float
    method: str  # "baseline", "diloco", "crdt"
    extra: dict = field(default_factory=dict)


def make_batch(config: SimConfig, device: torch.device) -> torch.Tensor:
    return torch.randint(0, config.model_config.vocab_size,
                         (config.batch_size, config.seq_len), device=device)


def eval_loss(model: nn.Module, config: SimConfig, device: torch.device, n_batches: int = 5) -> float:
    """Evaluate loss on random data (proxy for actual eval)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = make_batch(config, device)
            total += model.compute_loss(batch).item()
    model.train()
    return total / n_batches


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: SimConfig,
    device: torch.device,
    n_steps: int,
    global_step: int,
) -> tuple[list[float], int]:
    """Train for n_steps, return per-step losses and updated global_step."""
    model.train()
    losses = []
    for i in range(n_steps):
        lr = cosine_lr_schedule(global_step, config.warmup_steps, config.total_steps,
                                config.inner_lr, config.min_inner_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        batch = make_batch(config, device)
        loss = model.compute_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        global_step += 1
    return losses, global_step


# ─── Baseline: standard single-GPU training ───


def run_baseline(config: SimConfig) -> list[StepRecord]:
    logger.info("=== BASELINE: Standard training ===")
    device = torch.device(config.device)
    model = DistrainTransformer(config.model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.inner_lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    records: list[StepRecord] = []
    global_step = 0
    eval_every = max(1, config.total_steps // 40)

    while global_step < config.total_steps:
        chunk = min(eval_every, config.total_steps - global_step)
        losses, global_step = train_steps(model, optimizer, config, device, chunk, global_step)
        loss = eval_loss(model, config, device)
        records.append(StepRecord(step=global_step, loss=loss, method="baseline"))
        logger.info(f"  [baseline] step={global_step:5d}  eval_loss={loss:.4f}")

    return records


# ─── DiLoCo: round-based synchronous ───


def run_diloco(config: SimConfig) -> list[StepRecord]:
    logger.info("=== DILOCO: Round-based training ===")
    device = torch.device(config.device)

    # Master checkpoint
    master = DistrainTransformer(config.model_config).to(device)
    master_state = {n: p.data.clone() for n, p in master.named_parameters()}
    outer_opt = NesterovOuterOptimizer(lr=config.outer_lr, momentum=config.outer_momentum)

    records: list[StepRecord] = []
    total_steps_done = 0
    steps_per_node_per_round = config.h_diloco
    steps_per_round = steps_per_node_per_round * config.num_nodes
    num_rounds = max(1, config.total_steps // steps_per_round)

    for round_id in range(num_rounds):
        if total_steps_done >= config.total_steps:
            break

        outer_grads: list[dict[str, torch.Tensor]] = []

        for node_id in range(config.num_nodes):
            # Each node starts from master checkpoint
            node_model = DistrainTransformer(config.model_config).to(device)
            node_model.load_state_dict(master_state)
            node_opt = torch.optim.AdamW(node_model.parameters(), lr=config.inner_lr,
                                         betas=(0.9, 0.95), weight_decay=0.1)

            theta_start = {n: p.data.clone() for n, p in node_model.named_parameters()}
            _, total_steps_done_tmp = train_steps(
                node_model, node_opt, config, device,
                steps_per_node_per_round, total_steps_done,
            )

            # Outer gradient
            outer_grad = {n: theta_start[n] - p.data for n, p in node_model.named_parameters()}
            outer_grads.append(outer_grad)

        total_steps_done += steps_per_round

        # Average outer gradients
        avg_grad = {}
        for name in outer_grads[0]:
            avg_grad[name] = sum(g[name] for g in outer_grads) / len(outer_grads)

        # Apply outer optimizer
        master_state = outer_opt.step(
            {n: t.clone() for n, t in master_state.items()}, avg_grad,
        )

        # Eval
        master.load_state_dict(master_state)
        loss = eval_loss(master, config, device)
        records.append(StepRecord(
            step=total_steps_done, loss=loss, method="diloco",
            extra={"round": round_id, "grad_norm": sum(g.norm().item() for g in avg_grad.values())},
        ))
        logger.info(f"  [diloco] round={round_id:3d}  total_steps={total_steps_done:5d}  "
                    f"eval_loss={loss:.4f}")

    return records


# ─── CRDT: continuous asynchronous merge ───


def run_crdt(config: SimConfig) -> list[StepRecord]:
    logger.info("=== CRDT: Continuous merge training ===")
    device = torch.device(config.device)

    # Master checkpoint
    master = DistrainTransformer(config.model_config).to(device)
    master_state = {n: p.data.clone() for n, p in master.named_parameters()}
    outer_opt = NesterovOuterOptimizer(lr=config.outer_lr, momentum=config.outer_momentum)
    accumulator = GradientAccumulator(
        min_contributions=config.min_contributions,
        staleness_decay=config.staleness_decay,
        max_staleness=10,
    )

    # Per-node state: each node has its own model, optimizer, and checkpoint version
    node_models: list[nn.Module] = []
    node_optimizers: list[torch.optim.Optimizer] = []
    node_checkpoint_versions: list[int] = []

    for _ in range(config.num_nodes):
        m = DistrainTransformer(config.model_config).to(device)
        m.load_state_dict(master_state)
        o = torch.optim.AdamW(m.parameters(), lr=config.inner_lr,
                              betas=(0.9, 0.95), weight_decay=0.1)
        node_models.append(m)
        node_optimizers.append(o)
        node_checkpoint_versions.append(0)

    records: list[StepRecord] = []
    checkpoint_version = 0
    total_steps_done = 0
    seq_nums = [0] * config.num_nodes

    # Event queue: (priority, node_id) — lower priority = sooner
    # Priority is "simulated time" when node finishes its H_mini steps
    speeds = config.node_speeds[:config.num_nodes]
    while len(speeds) < config.num_nodes:
        speeds.append(0.5)

    event_queue: list[tuple[float, int]] = []
    for i, speed in enumerate(speeds):
        heapq.heappush(event_queue, (config.h_mini / speed, i))

    eval_interval_steps = max(config.h_mini * config.num_nodes,
                              config.total_steps // 40)
    last_eval_step = 0

    while total_steps_done < config.total_steps:
        sim_time, node_id = heapq.heappop(event_queue)
        speed = speeds[node_id]

        node_model = node_models[node_id]
        node_opt = node_optimizers[node_id]

        # Snapshot before training
        theta_start = {n: p.data.clone() for n, p in node_model.named_parameters()}

        # Train H_mini steps
        actual_steps = min(config.h_mini, config.total_steps - total_steps_done)
        if actual_steps <= 0:
            break

        _, _ = train_steps(node_model, node_opt, config, device, actual_steps, total_steps_done)
        total_steps_done += actual_steps

        # Compute outer gradient
        outer_grad = {n: theta_start[n] - p.data for n, p in node_model.named_parameters()}

        # Push to accumulator
        seq_nums[node_id] += 1
        accumulator.apply(
            node_id=f"node_{node_id}",
            delta=outer_grad,
            seq_num=seq_nums[node_id],
            inner_steps=actual_steps,
            checkpoint_version=node_checkpoint_versions[node_id],
            current_version=checkpoint_version,
        )

        # Check if time to produce checkpoint
        if accumulator.should_checkpoint():
            avg_grad = accumulator.read()
            master_state = outer_opt.step(
                {n: t.clone() for n, t in master_state.items()}, avg_grad,
            )
            checkpoint_version += 1
            accumulator.reset()

            # All nodes eventually pull new checkpoint
            # For simulation: nodes that are "fast" pull sooner, but we simplify
            # by having all nodes pull immediately when they next become active
            for i in range(config.num_nodes):
                node_models[i].load_state_dict(master_state)
                # Reset optimizer state for clean start
                node_optimizers[i] = torch.optim.AdamW(
                    node_models[i].parameters(), lr=config.inner_lr,
                    betas=(0.9, 0.95), weight_decay=0.1,
                )
                node_checkpoint_versions[i] = checkpoint_version

        # Schedule next push for this node
        next_time = sim_time + config.h_mini / speed
        heapq.heappush(event_queue, (next_time, node_id))

        # Periodic eval
        if total_steps_done - last_eval_step >= eval_interval_steps:
            master.load_state_dict(master_state)
            loss = eval_loss(master, config, device)
            records.append(StepRecord(
                step=total_steps_done, loss=loss, method="crdt",
                extra={"checkpoint_version": checkpoint_version,
                       "contributions": accumulator.num_contributions},
            ))
            logger.info(f"  [crdt] steps={total_steps_done:5d}  ckpt_v={checkpoint_version:3d}  "
                        f"eval_loss={loss:.4f}")
            last_eval_step = total_steps_done

    # Final eval
    master.load_state_dict(master_state)
    loss = eval_loss(master, config, device)
    records.append(StepRecord(step=total_steps_done, loss=loss, method="crdt",
                              extra={"checkpoint_version": checkpoint_version, "final": True}))
    logger.info(f"  [crdt] FINAL  steps={total_steps_done:5d}  ckpt_v={checkpoint_version:3d}  "
                f"eval_loss={loss:.4f}")
    return records


# ─── Main ───


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate CRDT continuous merge")
    parser.add_argument("--model-size", default=None, choices=list(CONFIGS.keys()),
                        help="Use a preset model (overrides tiny default)")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--h-diloco", type=int, default=500)
    parser.add_argument("--h-mini", type=int, default=50)
    parser.add_argument("--min-contributions", type=int, default=4)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/continuous_sim")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-diloco", action="store_true")
    args = parser.parse_args()

    config = SimConfig(
        num_nodes=args.nodes,
        total_steps=args.total_steps,
        h_diloco=args.h_diloco,
        h_mini=args.h_mini,
        min_contributions=args.min_contributions,
        outer_lr=args.outer_lr,
        inner_lr=args.inner_lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.model_size:
        config.model_config = CONFIGS[args.model_size]
        config.seq_len = min(64, config.model_config.max_seq_len)

    # Adjust node speeds to match node count
    config.node_speeds = config.node_speeds[:config.num_nodes]
    while len(config.node_speeds) < config.num_nodes:
        config.node_speeds.append(0.5)

    logger.info(f"Config: {config.num_nodes} nodes, {config.total_steps} total steps, "
                f"H_diloco={config.h_diloco}, H_mini={config.h_mini}")
    logger.info(f"Model: {config.model_config.param_count_str()}")
    logger.info(f"Node speeds: {config.node_speeds}")

    t0 = time.time()
    all_records: list[StepRecord] = []

    # Run all three methods
    if not args.skip_baseline:
        baseline_records = run_baseline(config)
        all_records.extend(baseline_records)

    if not args.skip_diloco:
        diloco_records = run_diloco(config)
        all_records.extend(diloco_records)

    crdt_records = run_crdt(config)
    all_records.extend(crdt_records)

    elapsed = time.time() - t0

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "num_nodes": config.num_nodes,
            "total_steps": config.total_steps,
            "h_diloco": config.h_diloco,
            "h_mini": config.h_mini,
            "min_contributions": config.min_contributions,
            "outer_lr": config.outer_lr,
            "inner_lr": config.inner_lr,
            "model_params": config.model_config.param_count_str(),
            "node_speeds": config.node_speeds,
        },
        "records": [{"step": r.step, "loss": r.loss, "method": r.method, **r.extra}
                    for r in all_records],
        "elapsed_seconds": elapsed,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    methods = {}
    for r in all_records:
        if r.method not in methods:
            methods[r.method] = []
        methods[r.method].append(r)

    print(f"\n{'Method':<15} {'Final Loss':>12} {'Steps':>10} {'Checkpoints':>14}")
    print("-" * 55)

    baseline_final = None
    for method_name, recs in methods.items():
        final = recs[-1]
        ckpt_info = ""
        if method_name == "diloco":
            ckpt_info = str(final.extra.get("round", "?"))
        elif method_name == "crdt":
            ckpt_info = str(final.extra.get("checkpoint_version", "?"))
        else:
            ckpt_info = "N/A"

        if method_name == "baseline":
            baseline_final = final.loss

        print(f"{method_name:<15} {final.loss:>12.4f} {final.step:>10d} {ckpt_info:>14}")

    if baseline_final:
        print(f"\n{'Method':<15} {'vs Baseline':>12}")
        print("-" * 30)
        for method_name, recs in methods.items():
            final = recs[-1]
            gap = ((final.loss / baseline_final) - 1) * 100
            marker = "✓" if abs(gap) < 15 else "✗"
            print(f"{method_name:<15} {gap:>+10.1f}%  {marker}")

    print(f"\nTotal simulation time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
