#!/usr/bin/env python3
"""End-to-end integration test for the Distrain distributed training system.

Starts MinIO (via docker compose), coordinator, and 3 node clients.
Verifies that checkpoint versions advance and the system functions end-to-end.

Usage:
    python scripts/integration_test.py
    python scripts/integration_test.py --target-versions 10 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
import torch
from botocore.config import Config as BotoConfig

# Add python/ to path so we can import distrain
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from distrain.model.config import ModelConfig
from distrain.model.transformer import DistrainTransformer
from distrain.training.checkpointing import save_checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "distrain-training"

COORDINATOR_PORT = 8000
COORDINATOR_URL = f"http://localhost:{COORDINATOR_PORT}"

# Tiny model for fast E2E testing
TINY_CONFIG = ModelConfig(
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    ffn_hidden_dim=128,
    vocab_size=256,
    max_seq_len=64,
)

# Coordinator binary
COORDINATOR_BIN = str(PROJECT_ROOT / "target" / "debug" / "coordinator")
NODE_BIN = str(PROJECT_ROOT / "target" / "debug" / "distrain-node")


def get_s3_client():
    """Create boto3 S3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4"),
    )


def wait_for_minio(timeout: int = 30) -> None:
    """Wait for MinIO to be ready."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{MINIO_ENDPOINT}/minio/health/live", timeout=2)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    raise TimeoutError("MinIO did not become ready")


def wait_for_coordinator(timeout: int = 15) -> None:
    """Wait for coordinator to be ready."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{COORDINATOR_URL}/health", timeout=2)
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError("Coordinator did not become ready")


def get_status() -> dict:
    """Get training status from coordinator."""
    import urllib.request

    resp = urllib.request.urlopen(f"{COORDINATOR_URL}/status", timeout=5)
    return json.loads(resp.read())


def bootstrap_minio(s3, tmp_dir: Path) -> None:
    """Create bucket and upload initial checkpoint."""
    # Create bucket
    try:
        s3.create_bucket(Bucket=BUCKET)
        print(f"  Created bucket: {BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists: {BUCKET}")

    # Create tiny model and save as v0 checkpoint
    model = DistrainTransformer(TINY_CONFIG)
    ckpt_path = tmp_dir / "v0_model.safetensors"
    save_checkpoint(model.state_dict(), ckpt_path)

    # Upload to checkpoints/v0/model.safetensors
    s3.upload_file(
        str(ckpt_path),
        BUCKET,
        "checkpoints/v0/model.safetensors",
    )
    print(f"  Uploaded initial checkpoint ({ckpt_path.stat().st_size} bytes)")


def write_node_config(path: Path, node_name: str) -> None:
    """Write a node TOML config file."""
    cache_dir = str(path.parent / "cache" / node_name)
    config = f"""coordinator_url = "{COORDINATOR_URL}"
api_key = ""
gpu_device = -1
target_push_interval_secs = 0.5
min_inner_steps = 5
max_inner_steps = 10
cache_dir = "{cache_dir}"
max_cache_gb = 1

[storage]
endpoint = "{MINIO_ENDPOINT}"
bucket = "{BUCKET}"
access_key_id = "{MINIO_ACCESS_KEY}"
secret_access_key = "{MINIO_SECRET_KEY}"
region = "us-east-1"
"""
    path.write_text(config.strip())


def start_coordinator(tmp_dir: Path, log_dir: Path) -> subprocess.Popen:
    """Start the coordinator process."""
    log_file = open(log_dir / "coordinator.log", "w")
    env = {
        **os.environ,
        "PORT": str(COORDINATOR_PORT),
        "HOST": "0.0.0.0",
        "R2_ENDPOINT": MINIO_ENDPOINT,
        "R2_BUCKET": BUCKET,
        "R2_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
        "R2_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
        "MIN_CONTRIBUTIONS": "3",
        "RUST_LOG": "info",
    }
    proc = subprocess.Popen(
        [COORDINATOR_BIN],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"  Coordinator PID: {proc.pid}")
    return proc


def start_node(name: str, config_path: Path, log_dir: Path) -> subprocess.Popen:
    """Start a node client process."""
    log_file = open(log_dir / f"{name}.log", "w")
    env = {
        **os.environ,
        "RUST_LOG": "info",
    }
    proc = subprocess.Popen(
        [NODE_BIN, "start", "--config", str(config_path)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"  {name} PID: {proc.pid}")
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Distrain integration test")
    parser.add_argument("--target-versions", type=int, default=5,
                        help="Target checkpoint versions to produce")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout in seconds")
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--keep-logs", action="store_true",
                        help="Don't delete log files on success")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="distrain_e2e_"))
    log_dir = tmp_dir / "logs"
    log_dir.mkdir()
    print(f"Working directory: {tmp_dir}")
    print(f"Logs: {log_dir}")

    processes: list[subprocess.Popen] = []
    docker_started = False
    success = False

    def cleanup():
        print("\nCleaning up...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
        if docker_started:
            subprocess.run(
                ["docker", "compose", "down", "-v"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )

    # Handle Ctrl+C
    def sigint_handler(sig, frame):
        cleanup()
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        # --- Check prerequisites ---
        print("\n=== Checking prerequisites ===")
        if not Path(COORDINATOR_BIN).exists():
            print(f"ERROR: Coordinator binary not found at {COORDINATOR_BIN}")
            print("Run: cargo build")
            sys.exit(1)
        if not Path(NODE_BIN).exists():
            print(f"ERROR: Node binary not found at {NODE_BIN}")
            print("Run: cargo build")
            sys.exit(1)
        print("  All binaries found")

        # --- Start MinIO ---
        print("\n=== Starting MinIO ===")
        subprocess.run(
            ["docker", "compose", "up", "-d", "minio"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
        )
        docker_started = True
        print("  MinIO container started, waiting for readiness...")
        wait_for_minio()
        print("  MinIO ready")

        # --- Bootstrap ---
        print("\n=== Bootstrapping ===")
        s3 = get_s3_client()
        bootstrap_minio(s3, tmp_dir)

        # --- Write node configs ---
        print("\n=== Writing node configs ===")
        node_configs = {}
        for i in range(args.num_nodes):
            name = f"node_{i}"
            config_path = tmp_dir / f"{name}.toml"
            write_node_config(config_path, name)
            node_configs[name] = config_path
            # Create cache dir
            (tmp_dir / "cache" / name).mkdir(parents=True, exist_ok=True)
        print(f"  Created {args.num_nodes} node configs")

        # --- Start coordinator ---
        print("\n=== Starting coordinator ===")
        coord_proc = start_coordinator(tmp_dir, log_dir)
        processes.append(coord_proc)
        wait_for_coordinator()
        print("  Coordinator ready")

        # --- Start nodes ---
        print(f"\n=== Starting {args.num_nodes} nodes ===")
        for name, config_path in node_configs.items():
            node_proc = start_node(name, config_path, log_dir)
            processes.append(node_proc)
            time.sleep(0.5)  # stagger starts slightly
        print("  All nodes started")

        # --- Monitor progress ---
        print(f"\n=== Monitoring (target: v{args.target_versions}, timeout: {args.timeout}s) ===")
        t0 = time.time()
        last_version = 0
        poll_interval = 2.0

        while time.time() - t0 < args.timeout:
            time.sleep(poll_interval)

            # Check if any process died
            for p in processes:
                if p.poll() is not None:
                    # Process exited
                    name = "coordinator" if p == coord_proc else "node"
                    print(f"\n  WARNING: {name} (PID {p.pid}) exited with code {p.returncode}")

            try:
                status = get_status()
                version = status.get("checkpoint_version", 0)
                contribs = status.get("accumulator_contributions", 0)
                active = status.get("active_nodes", 0)

                if version != last_version:
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:.0f}s] Checkpoint v{version} | "
                          f"active_nodes={active} | "
                          f"accumulator={contribs}")
                    last_version = version

                if version >= args.target_versions:
                    print(f"\n  Target reached: v{version} >= v{args.target_versions}")
                    break
            except Exception:
                pass  # coordinator might be busy during aggregation

        # Give a moment for final writes
        time.sleep(2)

        # --- Verify results ---
        print("\n=== Verification ===")
        final_status = get_status()
        final_version = final_status.get("checkpoint_version", 0)
        print(f"  Final checkpoint version: {final_version}")
        print(f"  Active nodes: {final_status.get('active_nodes', 0)}")
        print(f"  Total contributions: {final_status.get('total_contributions', 0)}")

        # Check that checkpoints exist in MinIO
        checkpoints_found = 0
        for v in range(1, final_version + 1):
            key = f"checkpoints/v{v}/model.safetensors"
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
                checkpoints_found += 1
            except Exception:
                print(f"  WARNING: Checkpoint v{v} not found in MinIO")
        print(f"  Checkpoints in MinIO: {checkpoints_found}/{final_version}")

        # Check metadata files
        metadata_found = 0
        for v in range(1, final_version + 1):
            key = f"checkpoints/v{v}/metadata.json"
            try:
                obj = s3.get_object(Bucket=BUCKET, Key=key)
                meta = json.loads(obj["Body"].read())
                metadata_found += 1
                if v <= 3:  # print first few
                    print(f"    v{v}: accepted={meta.get('contributions_accepted')}, "
                          f"grad_norm={meta.get('outer_grad_norm', 0):.4f}")
            except Exception:
                pass
        print(f"  Metadata files in MinIO: {metadata_found}/{final_version}")

        # Check optimizer states
        opt_found = 0
        for v in range(1, final_version + 1):
            key = f"optimizer_state/v{v}/velocity.safetensors"
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
                opt_found += 1
            except Exception:
                pass
        print(f"  Optimizer states in MinIO: {opt_found}/{final_version}")

        # --- Final verdict ---
        elapsed_total = time.time() - t0
        print(f"\n=== Results ({elapsed_total:.0f}s elapsed) ===")

        checks = []

        # Check 1: At least target versions produced
        check1 = final_version >= args.target_versions
        checks.append(check1)
        status1 = "PASS" if check1 else "FAIL"
        print(f"  [{status1}] Checkpoint versions: {final_version} >= {args.target_versions}")

        # Check 2: All checkpoints exist in MinIO
        check2 = checkpoints_found == final_version
        checks.append(check2)
        status2 = "PASS" if check2 else "FAIL"
        print(f"  [{status2}] All checkpoints in storage: {checkpoints_found}/{final_version}")

        # Check 3: Metadata files exist
        check3 = metadata_found == final_version
        checks.append(check3)
        status3 = "PASS" if check3 else "FAIL"
        print(f"  [{status3}] All metadata files: {metadata_found}/{final_version}")

        # Check 4: Multiple nodes contributed
        check4 = final_status.get("active_nodes", 0) >= 2
        checks.append(check4)
        status4 = "PASS" if check4 else "FAIL"
        print(f"  [{status4}] Multiple active nodes: {final_status.get('active_nodes', 0)}")

        success = all(checks)
        if success:
            print("\n  ALL CHECKS PASSED")
        else:
            print("\n  SOME CHECKS FAILED")
            print(f"  Check logs at: {log_dir}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

        if not success or args.keep_logs:
            print(f"\nLogs preserved at: {log_dir}")
            # Print tail of coordinator log for debugging
            coord_log = log_dir / "coordinator.log"
            if coord_log.exists():
                print(f"\n--- Last 30 lines of coordinator.log ---")
                lines = coord_log.read_text().splitlines()
                for line in lines[-30:]:
                    print(f"  {line}")
            # Print tail of first node log
            node_log = log_dir / "node_0.log"
            if node_log.exists():
                print(f"\n--- Last 20 lines of node_0.log ---")
                lines = node_log.read_text().splitlines()
                for line in lines[-20:]:
                    print(f"  {line}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
