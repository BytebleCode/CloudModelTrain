"""
Dataset integrity validation.

Checks:
  1. File exists and is non-empty
  2. Every line is valid JSON (for JSONL)
  3. Required fields present per agent formatting spec
  4. Optional SHA-256 checksum
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


def validate_jsonl(
    path: Path,
    required_fields: list[str] | None = None,
    expected_sha256: str | None = None,
) -> int:
    """
    Validate a JSONL file.

    Returns:
        Number of valid records.

    Raises:
        ValidationError on any failure.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise ValidationError(f"File is empty: {path}")

    # Optional SHA-256
    if expected_sha256:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual != expected_sha256:
            raise ValidationError(
                f"SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual}"
            )
        logger.info("SHA-256 verified: %s", path)

    # Parse every line
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Invalid JSON at {path}:{lineno}: {e}"
                )
            if required_fields:
                missing = [k for k in required_fields if k not in record]
                if missing:
                    raise ValidationError(
                        f"Missing fields {missing} at {path}:{lineno}"
                    )
            count += 1

    if count == 0:
        raise ValidationError(f"No records found in {path}")

    logger.info("Validated %d records in %s", count, path)
    return count
