#!/usr/bin/env python3
"""Download, tokenize, and shard a dataset for Distrain training.

Supports HuggingFace datasets (FineWeb-Edu, SmolLM-Corpus, etc.) and
converts them into the binary shard format (flat uint16 token IDs) used
by the Rust node client's data loader.

Usage:
    # Quick test shards (random data, no download)
    python scripts/prepare_data.py test --output-dir data/test_shards

    # FineWeb-Edu 10BT sample (~10B tokens, best for 125M model)
    python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb

    # FineWeb-Edu 100BT sample
    python scripts/prepare_data.py fineweb-edu-100bt --output-dir data/fineweb_100bt

    # SmolLM-Corpus Cosmopedia v2 (~28B tokens, synthetic textbooks)
    python scripts/prepare_data.py smollm-cosmopedia --output-dir data/cosmopedia

    # Any HuggingFace dataset with a "text" column
    python scripts/prepare_data.py hf --dataset "HuggingFaceFW/fineweb-edu" \
        --config "sample-10BT" --output-dir data/custom

    # Upload shards to MinIO / R2 after preparation
    python scripts/prepare_data.py fineweb-edu-10bt --output-dir data/fineweb \
        --upload --s3-endpoint http://localhost:9000 --s3-bucket distrain-training

Requirements:
    pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

TOKENS_PER_SHARD_DEFAULT = 10_000_000  # 10M tokens per shard = ~20 MB


def write_shard(tokens: np.ndarray, path: Path) -> None:
    """Write a uint16 token array to a binary shard file."""
    assert tokens.dtype == np.uint16
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)


def write_manifest(shard_paths: list[Path], vocab_size: int, output_dir: Path) -> None:
    """Write shard manifest JSON."""
    manifest = {
        "num_shards": len(shard_paths),
        "vocab_size": vocab_size,
        "shards": [
            {
                "filename": p.name,
                "num_tokens": p.stat().st_size // 2,
                "size_bytes": p.stat().st_size,
            }
            for p in shard_paths
        ],
        "total_tokens": sum(p.stat().st_size // 2 for p in shard_paths),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = manifest["total_tokens"]
    size_mb = sum(p.stat().st_size for p in shard_paths) / 1e6
    logger.info(f"Manifest: {len(shard_paths)} shards, {total:,} tokens, {size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def get_tokenizer(vocab_size: int = 32768):
    """Get the Mistral v0.3 tokenizer (32768 vocab, no modulo mapping needed)."""
    from tokenizers import Tokenizer

    tokenizer_path = PROJECT_ROOT / "tokenizers" / "mistral-v0.3.json"
    enc = Tokenizer.from_file(str(tokenizer_path))
    assert enc.get_vocab_size() == vocab_size, (
        f"Tokenizer vocab size {enc.get_vocab_size()} != expected {vocab_size}"
    )
    return enc, vocab_size


def tokenize_and_shard(
    texts,
    output_dir: Path,
    vocab_size: int = 32768,
    tokens_per_shard: int = TOKENS_PER_SHARD_DEFAULT,
) -> list[Path]:
    """Tokenize an iterable of texts and write as binary shards.

    Uses the Mistral v0.3 tokenizer which has exactly 32768 tokens —
    IDs are natively in [0, 32768), no modulo mapping needed.
    """
    enc, _ = get_tokenizer(vocab_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer: list[int] = []
    paths: list[Path] = []

    def flush_shard():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        arr = np.array(buffer[:tokens_per_shard], dtype=np.uint16)
        path = output_dir / f"shard_{shard_idx:04d}.bin"
        write_shard(arr, path)
        paths.append(path)
        logger.info(f"  shard_{shard_idx:04d}.bin — {len(arr):,} tokens")
        shard_idx += 1
        buffer = buffer[tokens_per_shard:]

    for text in texts:
        if not text or not text.strip():
            continue
        tokens = enc.encode(text).ids
        buffer.extend(tokens)
        while len(buffer) >= tokens_per_shard:
            flush_shard()

    # Final partial shard (only if substantial)
    if len(buffer) >= tokens_per_shard // 10:
        arr = np.array(buffer, dtype=np.uint16)
        path = output_dir / f"shard_{shard_idx:04d}.bin"
        write_shard(arr, path)
        paths.append(path)
        logger.info(f"  shard_{shard_idx:04d}.bin — {len(arr):,} tokens (final)")
    elif buffer:
        logger.info(f"  Discarded {len(buffer):,} trailing tokens (too few for a shard)")

    return paths


# ---------------------------------------------------------------------------
# Dataset modes
# ---------------------------------------------------------------------------


def mode_test(args) -> None:
    """Create random test shards (no download, no tokenizer)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(args.num_shards):
        tokens = np.random.randint(0, args.vocab_size, size=args.tokens_per_shard, dtype=np.uint16)
        path = output_dir / f"shard_{i:04d}.bin"
        write_shard(tokens, path)
        paths.append(path)
        logger.info(f"  {path.name} — {args.tokens_per_shard:,} random tokens")
    write_manifest(paths, args.vocab_size, output_dir)


def mode_hf_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    text_column: str,
    output_dir: Path,
    vocab_size: int,
    tokens_per_shard: int,
    max_samples: int | None = None,
    streaming: bool = True,
) -> None:
    """Download a HuggingFace dataset and tokenize into shards."""
    from datasets import load_dataset

    logger.info(f"Loading {dataset_name}" + (f" ({config_name})" if config_name else ""))

    ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)

    if max_samples:
        ds = ds.take(max_samples)
        logger.info(f"  Limited to {max_samples:,} samples")

    def text_iter():
        count = 0
        for example in ds:
            text = example.get(text_column, "")
            if text:
                yield text
                count += 1
                if count % 100_000 == 0:
                    logger.info(f"  Processed {count:,} documents...")

    paths = tokenize_and_shard(text_iter(), output_dir, vocab_size, tokens_per_shard)
    write_manifest(paths, vocab_size, output_dir)


def mode_fineweb_edu_10bt(args) -> None:
    """FineWeb-Edu sample-10BT — best for 125M model training."""
    mode_hf_dataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        text_column="text",
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        tokens_per_shard=args.tokens_per_shard,
        max_samples=args.max_samples,
    )


def mode_fineweb_edu_100bt(args) -> None:
    """FineWeb-Edu sample-100BT — for longer training runs."""
    mode_hf_dataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        config_name="sample-100BT",
        split="train",
        text_column="text",
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        tokens_per_shard=args.tokens_per_shard,
        max_samples=args.max_samples,
    )


def mode_smollm_cosmopedia(args) -> None:
    """SmolLM-Corpus Cosmopedia v2 — synthetic textbooks, ~28B tokens."""
    mode_hf_dataset(
        dataset_name="HuggingFaceTB/smollm-corpus",
        config_name="cosmopedia-v2",
        split="train",
        text_column="text",
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        tokens_per_shard=args.tokens_per_shard,
        max_samples=args.max_samples,
    )


def mode_hf_custom(args) -> None:
    """Any HuggingFace dataset with a text column."""
    mode_hf_dataset(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        text_column=args.text_column,
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        tokens_per_shard=args.tokens_per_shard,
        max_samples=args.max_samples,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_to_s3(output_dir: Path, endpoint: str, bucket: str, access_key: str, secret_key: str) -> None:
    """Upload shards and manifest to S3/MinIO/R2."""
    import boto3
    from botocore.config import Config as BotoConfig

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4"),
    )

    shard_files = sorted(output_dir.glob("shard_*.bin"))
    manifest_file = output_dir / "manifest.json"

    for path in shard_files:
        key = f"data/{path.name}"
        logger.info(f"  Uploading {path.name} -> s3://{bucket}/{key}")
        s3.upload_file(str(path), bucket, key)

    if manifest_file.exists():
        logger.info(f"  Uploading manifest.json -> s3://{bucket}/data/manifest.json")
        s3.upload_file(str(manifest_file), bucket, "data/manifest.json")

    logger.info("Upload complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Distrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Common args added to each subcommand
    def add_common(p):
        p.add_argument("--output-dir", required=True, help="Directory to write shards")
        p.add_argument("--vocab-size", type=int, default=32768, help="Model vocabulary size")
        p.add_argument("--tokens-per-shard", type=int, default=TOKENS_PER_SHARD_DEFAULT, help="Tokens per shard file")
        p.add_argument("--upload", action="store_true", help="Upload shards to S3/MinIO/R2")
        p.add_argument("--s3-endpoint", default="http://localhost:9000")
        p.add_argument("--s3-bucket", default="distrain-training")
        p.add_argument("--s3-access-key", default="minioadmin")
        p.add_argument("--s3-secret-key", default="minioadmin")

    # test
    p_test = sub.add_parser("test", help="Random test shards (no download)")
    add_common(p_test)
    p_test.add_argument("--num-shards", type=int, default=4)

    # fineweb-edu-10bt
    p_fw10 = sub.add_parser("fineweb-edu-10bt", help="FineWeb-Edu 10BT sample (~10B tokens)")
    add_common(p_fw10)
    p_fw10.add_argument("--max-samples", type=int, default=None, help="Limit number of documents")

    # fineweb-edu-100bt
    p_fw100 = sub.add_parser("fineweb-edu-100bt", help="FineWeb-Edu 100BT sample (~100B tokens)")
    add_common(p_fw100)
    p_fw100.add_argument("--max-samples", type=int, default=None)

    # smollm-cosmopedia
    p_cosmo = sub.add_parser("smollm-cosmopedia", help="SmolLM Cosmopedia v2 (~28B tokens)")
    add_common(p_cosmo)
    p_cosmo.add_argument("--max-samples", type=int, default=None)

    # hf (generic)
    p_hf = sub.add_parser("hf", help="Any HuggingFace dataset")
    add_common(p_hf)
    p_hf.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    p_hf.add_argument("--config", default=None, help="Dataset config/subset name")
    p_hf.add_argument("--split", default="train")
    p_hf.add_argument("--text-column", default="text", help="Column containing text")
    p_hf.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    MODE_MAP = {
        "test": mode_test,
        "fineweb-edu-10bt": mode_fineweb_edu_10bt,
        "fineweb-edu-100bt": mode_fineweb_edu_100bt,
        "smollm-cosmopedia": mode_smollm_cosmopedia,
        "hf": mode_hf_custom,
    }

    logger.info(f"Mode: {args.mode}")
    MODE_MAP[args.mode](args)

    if args.upload:
        output_dir = Path(args.output_dir)
        logger.info(f"Uploading to {args.s3_endpoint}/{args.s3_bucket}")
        upload_to_s3(output_dir, args.s3_endpoint, args.s3_bucket, args.s3_access_key, args.s3_secret_key)

    logger.info("Done")


if __name__ == "__main__":
    main()
