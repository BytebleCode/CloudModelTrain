"""
Dataset puller — idempotent sync from remote storage to local NVMe cache.

Fully implements S3; other backends follow the same interface.
Can be run standalone:  python -m src.data.pull --agent code_writer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend: Local
# ---------------------------------------------------------------------------

def _resolve_local(uri: str, cache_dir: Path) -> Path:
    """If the dataset is already a local path, just return it."""
    p = Path(uri)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {uri}")
    logger.info("Using local dataset: %s", p)
    return p


# ---------------------------------------------------------------------------
# Backend: S3
# ---------------------------------------------------------------------------

def _resolve_s3(uri: str, cache_dir: Path) -> Path:
    """
    Download from S3 to *cache_dir* using boto3.
    Idempotent: skips files whose local size + ETag match remote.
    """
    import boto3
    from botocore.exceptions import ClientError

    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If the URI points to a single object (has extension), download it.
    # Otherwise treat as prefix and list all objects.
    if "." in prefix.split("/")[-1]:
        keys = [prefix]
    else:
        keys = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

    if not keys:
        raise FileNotFoundError(f"No objects found at {uri}")

    downloaded: list[Path] = []
    for key in keys:
        # Flatten into cache_dir keeping only the filename
        filename = key.split("/")[-1]
        local_path = cache_dir / filename

        # Idempotency: compare size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            remote_size = head["ContentLength"]
        except ClientError:
            raise FileNotFoundError(f"Cannot access s3://{bucket}/{key}")

        if local_path.exists() and local_path.stat().st_size == remote_size:
            logger.info("Cache hit (size match): %s", local_path)
            downloaded.append(local_path)
            continue

        logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
        s3.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)

    # Return the first JSONL file found, or the cache dir itself
    jsonl_files = [p for p in downloaded if p.suffix == ".jsonl"]
    if jsonl_files:
        return jsonl_files[0]
    return cache_dir


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {
    "s3": _resolve_s3,
    "": _resolve_local,    # no scheme = local path
    "file": _resolve_local,
}


def pull_dataset(uri: str, cache_dir: str | Path) -> Path:
    """
    Resolve a dataset URI to a local path, downloading if necessary.

    Args:
        uri: local path, s3://bucket/key, etc.
        cache_dir: local directory for caching remote datasets.

    Returns:
        Path to the local dataset file or directory.
    """
    cache_dir = Path(cache_dir)
    scheme = urlparse(uri).scheme

    backend = _BACKENDS.get(scheme)
    if backend is None:
        raise ValueError(
            f"Unsupported dataset URI scheme '{scheme}'. "
            f"Supported: {list(_BACKENDS.keys())}"
        )

    return backend(uri, cache_dir)


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

def main():
    from src.config import load_registry

    parser = argparse.ArgumentParser(description="Pull dataset for an agent")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    registry = load_registry()
    if args.agent not in registry:
        raise ValueError(f"Unknown agent: {args.agent}")

    agent_reg = registry[args.agent]
    cache_dir = args.cache_dir or str(Path("data_cache") / args.agent)

    uri = agent_reg["dataset_uri"]
    local_path = pull_dataset(uri, cache_dir)
    print(f"Dataset ready at: {local_path}")

    # Also pull eval if specified
    eval_uri = agent_reg.get("eval_uri", "")
    if eval_uri:
        eval_path = pull_dataset(eval_uri, cache_dir)
        print(f"Eval dataset ready at: {eval_path}")


if __name__ == "__main__":
    main()
