#!/usr/bin/env python3
"""End-to-end heterogeneous node test for Distrain.

Runs the full DiLoCo cycle with 3 heterogeneous node types:
  1. Native Mac node  — distrain-node binary
  2. Docker node      — same binary in a container
  3. WASM headless    — Node.js running wasm_node_headless.mjs

Uses the Tiny model (64 hidden, 2 layers, 256 vocab) with random data.
MIN_CONTRIBUTIONS=2 for fast iteration.

Usage:
    python scripts/e2e_heterogeneous.py
    python scripts/e2e_heterogeneous.py --target-versions 3 --timeout 120
    python scripts/e2e_heterogeneous.py --skip-docker   # skip Docker node
    python scripts/e2e_heterogeneous.py --skip-wasm     # skip WASM node
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

# Add python/ to path
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

COORDINATOR_BIN = str(PROJECT_ROOT / "target" / "debug" / "coordinator")
NODE_BIN = str(PROJECT_ROOT / "target" / "debug" / "distrain-node")
NODE_JS_BIN = "node"
WASM_SCRIPT = str(PROJECT_ROOT / "scripts" / "wasm_node_headless.mjs")


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
    try:
        s3.create_bucket(Bucket=BUCKET)
        print(f"  Created bucket: {BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists: {BUCKET}")

    # Create tiny model and save as v0 checkpoint
    model = DistrainTransformer(TINY_CONFIG)
    ckpt_path = tmp_dir / "v0_model.safetensors"
    save_checkpoint(model.state_dict(), ckpt_path)

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
        "MIN_CONTRIBUTIONS": "2",  # Lower for heterogeneous test
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


def start_native_node(name: str, config_path: Path, log_dir: Path) -> subprocess.Popen:
    """Start a native Rust node client."""
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


def start_docker_node(log_dir: Path) -> subprocess.Popen | None:
    """Start a Docker node via docker compose."""
    log_file = open(log_dir / "docker-node.log", "w")
    try:
        proc = subprocess.Popen(
            ["docker", "compose", "up", "--build", "node-docker"],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"  Docker node PID: {proc.pid}")
        return proc
    except FileNotFoundError:
        print("  WARNING: docker not found, skipping Docker node")
        return None


def start_wasm_node(log_dir: Path) -> subprocess.Popen | None:
    """Start a headless WASM node via Node.js."""
    # Check if WASM module is built
    wasm_pkg = PROJECT_ROOT / "web" / "pkg-node"
    if not wasm_pkg.exists():
        print("  WARNING: WASM package not built, skipping WASM node")
        print("           Build with: wasm-pack build crates/wasm --target nodejs --out-dir ../../web/pkg-node")
        return None

    log_file = open(log_dir / "wasm-node.log", "w")
    try:
        proc = subprocess.Popen(
            [
                NODE_JS_BIN, WASM_SCRIPT,
                "--coordinator", COORDINATOR_URL,
                "--steps", "5",
                "--rounds", "100",
                "--s3-endpoint", MINIO_ENDPOINT,
                "--bucket", BUCKET,
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"  WASM headless node PID: {proc.pid}")
        return proc
    except FileNotFoundError:
        print("  WARNING: node not found, skipping WASM node")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Distrain heterogeneous E2E test")
    parser.add_argument("--target-versions", type=int, default=3,
                        help="Target checkpoint versions")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip Docker node")
    parser.add_argument("--skip-wasm", action="store_true",
                        help="Skip WASM node")
    parser.add_argument("--keep-logs", action="store_true",
                        help="Preserve logs on success")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="distrain_e2e_hetero_"))
    log_dir = tmp_dir / "logs"
    log_dir.mkdir()
    print(f"Working directory: {tmp_dir}")
    print(f"Logs: {log_dir}")

    processes: list[subprocess.Popen] = []
    docker_started = False
    docker_node_started = False
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
        if docker_started or docker_node_started:
            subprocess.run(
                ["docker", "compose", "down", "-v"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )

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
        print("  Rust binaries found")

        node_types = ["native"]
        if not args.skip_docker:
            node_types.append("docker")
        if not args.skip_wasm:
            node_types.append("wasm")
        print(f"  Node types: {', '.join(node_types)}")

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

        # --- Start coordinator ---
        print("\n=== Starting coordinator (MIN_CONTRIBUTIONS=2) ===")
        coord_proc = start_coordinator(tmp_dir, log_dir)
        processes.append(coord_proc)
        wait_for_coordinator()
        print("  Coordinator ready")

        # --- Start native node ---
        print("\n=== Starting native Mac node ===")
        native_config = tmp_dir / "native_node.toml"
        write_node_config(native_config, "native")
        (tmp_dir / "cache" / "native").mkdir(parents=True, exist_ok=True)
        native_proc = start_native_node("native-mac", native_config, log_dir)
        processes.append(native_proc)

        # --- Start Docker node ---
        docker_proc = None
        if not args.skip_docker:
            print("\n=== Starting Docker node ===")
            docker_proc = start_docker_node(log_dir)
            if docker_proc:
                docker_node_started = True
                processes.append(docker_proc)
        else:
            print("\n=== Skipping Docker node ===")

        # --- Start WASM node ---
        wasm_proc = None
        if not args.skip_wasm:
            print("\n=== Starting WASM headless node ===")
            wasm_proc = start_wasm_node(log_dir)
            if wasm_proc:
                processes.append(wasm_proc)
        else:
            print("\n=== Skipping WASM node ===")

        # --- Monitor progress ---
        active_types = ["native"]
        if docker_proc:
            active_types.append("docker")
        if wasm_proc:
            active_types.append("wasm")

        print(f"\n=== Monitoring ({', '.join(active_types)}) ===")
        print(f"    Target: v{args.target_versions}, timeout: {args.timeout}s")

        t0 = time.time()
        last_version = 0
        poll_interval = 2.0

        while time.time() - t0 < args.timeout:
            time.sleep(poll_interval)

            # Check for dead processes
            for p in processes:
                if p.poll() is not None and p != docker_proc:
                    # Docker compose child may exit — that's normal for "up" mode
                    print(f"\n  WARNING: Process PID {p.pid} exited with code {p.returncode}")

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
                pass

        time.sleep(2)

        # --- Verification ---
        print("\n=== Verification ===")
        final_status = get_status()
        final_version = final_status.get("checkpoint_version", 0)
        print(f"  Final checkpoint version: {final_version}")
        print(f"  Active nodes: {final_status.get('active_nodes', 0)}")
        print(f"  Total contributions: {final_status.get('total_contributions', 0)}")

        # Check checkpoints in MinIO
        checkpoints_found = 0
        for v in range(1, final_version + 1):
            key = f"checkpoints/v{v}/model.safetensors"
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
                checkpoints_found += 1
            except Exception:
                print(f"  WARNING: Checkpoint v{v} not found")
        print(f"  Checkpoints in MinIO: {checkpoints_found}/{final_version}")

        # Check metadata
        metadata_found = 0
        for v in range(1, final_version + 1):
            key = f"checkpoints/v{v}/metadata.json"
            try:
                obj = s3.get_object(Bucket=BUCKET, Key=key)
                meta = json.loads(obj["Body"].read())
                metadata_found += 1
                if v <= 3:
                    print(f"    v{v}: accepted={meta.get('contributions_accepted')}, "
                          f"delta_norm={meta.get('outer_delta_norm', 0):.4f}")
            except Exception:
                pass
        print(f"  Metadata files: {metadata_found}/{final_version}")

        # Check deltas directory for multiple node types
        delta_keys = []
        try:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="deltas/", MaxKeys=100)
            delta_keys = [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception:
            pass

        node_types_seen = set()
        for key in delta_keys:
            fname = key.split("/")[-1]
            if fname.startswith("wasm-") or fname.startswith("browser-"):
                node_types_seen.add("wasm")
            elif fname.startswith("native") or fname.startswith("node_"):
                node_types_seen.add("native")
            else:
                node_types_seen.add("unknown")
        print(f"  Delta files: {len(delta_keys)}, node types: {node_types_seen or 'none'}")

        # --- Final verdict ---
        elapsed_total = time.time() - t0
        print(f"\n=== Results ({elapsed_total:.0f}s elapsed) ===")

        checks = []

        # Check 1: Target versions
        check1 = final_version >= args.target_versions
        checks.append(check1)
        s1 = "PASS" if check1 else "FAIL"
        print(f"  [{s1}] Checkpoint versions: {final_version} >= {args.target_versions}")

        # Check 2: All checkpoints exist
        check2 = checkpoints_found == final_version
        checks.append(check2)
        s2 = "PASS" if check2 else "FAIL"
        print(f"  [{s2}] All checkpoints in storage: {checkpoints_found}/{final_version}")

        # Check 3: Metadata exists
        check3 = metadata_found == final_version
        checks.append(check3)
        s3_status = "PASS" if check3 else "FAIL"
        print(f"  [{s3_status}] Metadata files: {metadata_found}/{final_version}")

        # Check 4: Multiple contributions
        total_contribs = final_status.get("total_contributions", 0)
        check4 = total_contribs >= 2
        checks.append(check4)
        s4 = "PASS" if check4 else "FAIL"
        print(f"  [{s4}] Contributions received: {total_contribs}")

        # Check 5: Non-zero delta norms (proves nodes compute real deltas)
        check5 = True  # optimistic
        for v in range(1, min(final_version + 1, 4)):
            try:
                obj = s3.get_object(
                    Bucket=BUCKET,
                    Key=f"checkpoints/v{v}/metadata.json",
                )
                meta = json.loads(obj["Body"].read())
                norm = meta.get("outer_delta_norm", 0)
                if norm < 1e-10:
                    check5 = False
                    print(f"  WARNING: v{v} delta norm near zero: {norm}")
            except Exception:
                pass
        checks.append(check5)
        s5 = "PASS" if check5 else "FAIL"
        print(f"  [{s5}] Delta norms non-zero (real training signal)")

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
            for log_name in ["coordinator.log", "native-mac.log", "docker-node.log", "wasm-node.log"]:
                log_path = log_dir / log_name
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    if lines:
                        print(f"\n--- Last 20 lines of {log_name} ---")
                        for line in lines[-20:]:
                            print(f"  {line}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
