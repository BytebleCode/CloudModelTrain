#!/usr/bin/env python3
"""
Build real Python training datasets for each agent from HuggingFace Hub.

Downloads, filters, reformats, and writes JSONL files to datasets/<agent>/train.jsonl.

Usage:
    python scripts/build_datasets.py                    # build all agents
    python scripts/build_datasets.py --agent code_writer  # build one agent
    python scripts/build_datasets.py --agent code_writer --max_samples 1000  # limit size
    python scripts/build_datasets.py --list             # show available builders

Each builder pulls a real, freely-available HuggingFace dataset and reformats it
into the JSONL schema expected by the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_datasets")

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def write_jsonl(records: list[dict], path: Path) -> int:
    """Write records to JSONL, return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


# ============================================================================
# code_writer: iamtarun/python_code_instructions_18k_alpaca
# ============================================================================

def build_code_writer(max_samples: int | None = None) -> int:
    """
    Python code generation from natural-language instructions.
    Source: 18k instruction/input/output triples, 100% Python.
    """
    from datasets import load_dataset

    logger.info("Loading iamtarun/python_code_instructions_18k_alpaca...")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

    records = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        # Skip very short outputs (likely incomplete)
        if len(output) < 20:
            continue
        records.append({
            "instruction": instruction,
            "input": inp,
            "output": output,
        })

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "code_writer" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("code_writer: %d records -> %s", count, path)
    return count


# ============================================================================
# test_generator: KAKA22/CodeRM-UnitTest
# ============================================================================

def build_test_generator(max_samples: int | None = None) -> int:
    """
    Unit test generation from problem descriptions + reference code.
    Source: CodeRM-UnitTest — problems with ground truth code and unit tests.
    """
    from datasets import load_dataset

    logger.info("Loading KAKA22/CodeRM-UnitTest...")
    try:
        ds = load_dataset("KAKA22/CodeRM-UnitTest", split="train")
    except Exception as e:
        logger.warning("CodeRM-UnitTest failed (%s), falling back to mbpp", e)
        return _build_test_generator_mbpp(max_samples)

    records = []
    for row in ds:
        question = (row.get("question") or "").strip()
        ground_truth = (row.get("code_ground_truth") or "").strip()
        unit_tests = row.get("unit_tests") or []

        if not question or not ground_truth or not unit_tests:
            continue

        # Combine unit tests into a single test block
        test_lines = []
        for ut in unit_tests[:5]:  # cap at 5 tests per problem
            code = ut.get("code", "").strip() if isinstance(ut, dict) else str(ut).strip()
            if code:
                test_lines.append(code)

        if not test_lines:
            continue

        test_block = "\n\n".join(test_lines)

        records.append({
            "instruction": "Generate comprehensive Python unit tests for the following code.",
            "input": f"## Problem\n{question}\n\n## Implementation\n```python\n{ground_truth}\n```",
            "output": test_block,
        })

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "test_generator" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("test_generator: %d records -> %s", count, path)
    return count


def _build_test_generator_mbpp(max_samples: int | None = None) -> int:
    """Fallback: use MBPP (Mostly Basic Python Problems) for test generation."""
    from datasets import load_dataset

    logger.info("Loading mbpp (fallback)...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")

    records = []
    for row in ds:
        text = (row.get("text") or "").strip()
        code = (row.get("code") or "").strip()
        tests = row.get("test_list") or []

        if not text or not code or not tests:
            continue

        test_block = "\n".join(tests)
        records.append({
            "instruction": "Generate Python test assertions for the following function.",
            "input": f"## Task\n{text}\n\n## Implementation\n```python\n{code}\n```",
            "output": test_block,
        })

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "test_generator" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("test_generator (mbpp fallback): %d records -> %s", count, path)
    return count


# ============================================================================
# static_reviewer: HuggingFaceH4/Code-Feedback (multi-turn code review)
# ============================================================================

def build_static_reviewer(max_samples: int | None = None) -> int:
    """
    Code review and feedback in multi-turn chat format.
    Source: Code-Feedback — 66k conversations about code debugging,
    improvement, and review. Filter to Python-related conversations.
    """
    from datasets import load_dataset

    logger.info("Loading HuggingFaceH4/Code-Feedback...")
    try:
        ds = load_dataset("HuggingFaceH4/Code-Feedback", split="train_sft")
    except Exception:
        ds = load_dataset("HuggingFaceH4/Code-Feedback", split="train")

    python_keywords = {"python", "def ", "import ", "class ", ".py", "pytest", "pip install"}
    records = []
    for row in ds:
        messages = row.get("messages") or []
        if len(messages) < 2:
            continue

        # Filter: at least one message must reference Python
        full_text = " ".join(m.get("content", "")[:500].lower() for m in messages)
        if not any(kw in full_text for kw in python_keywords):
            continue

        # Normalize role names
        cleaned_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role not in ("user", "assistant", "system"):
                role = "user" if role == "human" else "assistant"
            cleaned_messages.append({"role": role, "content": content})

        if len(cleaned_messages) < 2:
            continue

        records.append({"messages": cleaned_messages})

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "static_reviewer" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("static_reviewer: %d records -> %s", count, path)
    return count


# ============================================================================
# security_auditor: CyberNative/Code_Vulnerability_Security_DPO
# ============================================================================

def build_security_auditor(max_samples: int | None = None) -> int:
    """
    Security vulnerability detection and fixing in Python code.
    Source: Code_Vulnerability_Security_DPO — DPO pairs of secure vs vulnerable code.
    Filtered to Python.
    """
    from datasets import load_dataset

    logger.info("Loading CyberNative/Code_Vulnerability_Security_DPO...")
    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")

    records = []
    for row in ds:
        lang = (row.get("lang") or "").strip().lower()
        if lang not in ("python", "python3", "py"):
            continue

        vulnerability = (row.get("vulnerability") or "").strip()
        question = (row.get("question") or "").strip()
        chosen = (row.get("chosen") or "").strip()       # secure code
        rejected = (row.get("rejected") or "").strip()    # vulnerable code

        if not question or not chosen or not rejected:
            continue

        # Format as: given vulnerable code, identify issues and provide secure version
        instruction = (
            "Audit the following Python code for security vulnerabilities. "
            "Identify all issues with CWE references where applicable, "
            "explain the risks, and provide a secure implementation."
        )
        input_text = f"## Vulnerability Context\n{vulnerability}\n\n## Code to Audit\n```python\n{rejected}\n```"
        output_text = f"## Security Analysis\n{vulnerability}\n\n## Secure Implementation\n```python\n{chosen}\n```"

        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        })

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "security_auditor" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("security_auditor: %d records -> %s", count, path)
    return count


# ============================================================================
# performance_optimizer: CodeParrot/apps (competitive programming solutions)
# + HuggingFaceH4/Code-Feedback (optimization conversations)
# ============================================================================

def build_performance_optimizer(max_samples: int | None = None) -> int:
    """
    Code performance optimization for Python.
    Combines two strategies:
    1. Code-Feedback conversations filtered for optimization topics
    2. CodeContests / APPS solutions reformatted as optimization tasks
    """
    from datasets import load_dataset

    records = []

    # --- Strategy 1: Code-Feedback filtered for optimization ---
    logger.info("Loading HuggingFaceH4/Code-Feedback for optimization examples...")
    try:
        try:
            ds = load_dataset("HuggingFaceH4/Code-Feedback", split="train_sft")
        except Exception:
            ds = load_dataset("HuggingFaceH4/Code-Feedback", split="train")
        opt_keywords = {
            "optimize", "optimise", "faster", "slow", "performance", "efficient",
            "efficiency", "speed up", "time complexity", "space complexity",
            "memory usage", "big o", "bottleneck", "profil",
        }
        python_keywords = {"python", "def ", "import "}

        for row in ds:
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            first_msg = (messages[0].get("content") or "").lower()
            # Must be about optimization AND Python
            has_opt = any(kw in first_msg for kw in opt_keywords)
            has_python = any(kw in first_msg for kw in python_keywords)
            if not (has_opt and has_python):
                continue

            # Convert to instruction format
            user_content = (messages[0].get("content") or "").strip()
            assistant_parts = []
            for msg in messages[1:]:
                if msg.get("role") == "assistant":
                    assistant_parts.append((msg.get("content") or "").strip())

            if not assistant_parts:
                continue

            records.append({
                "instruction": "Analyze the following Python code for performance issues and provide an optimized version with explanation.",
                "input": user_content,
                "output": "\n\n".join(assistant_parts),
            })
    except Exception as e:
        logger.warning("Code-Feedback optimization extraction failed: %s", e)

    # Note: MBPP was previously used here but produced fake optimization pairs
    # (same code as input and output). Removed — only real optimization
    # conversations from Code-Feedback are used now.
    if not records:
        logger.warning("No optimization records found from Code-Feedback")

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "performance_optimizer" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("performance_optimizer: %d records -> %s", count, path)
    return count


# ============================================================================
# docs_generator: Nan-Do/code-search-net-python
# ============================================================================

def build_docs_generator(max_samples: int | None = None) -> int:
    """
    Python docstring and documentation generation.
    Source: code-search-net-python — 455k Python functions with docstrings.
    """
    from datasets import load_dataset

    logger.info("Loading Nan-Do/code-search-net-python...")
    ds = load_dataset("Nan-Do/code-search-net-python", split="train")

    records = []
    for row in ds:
        code = (row.get("code") or "").strip()
        docstring = (row.get("docstring") or "").strip()
        func_name = (row.get("func_name") or "").strip()

        if not code or not docstring:
            continue
        # Skip trivially short docstrings
        if len(docstring) < 20:
            continue
        # Skip very long functions (noise)
        if len(code) > 5000:
            continue

        # Remove existing docstring from code to create the "input"
        # (model should learn to generate it)
        code_without_doc = _strip_docstring(code)

        records.append({
            "instruction": (
                "Generate a comprehensive Python docstring for the following function. "
                "Include a description, Args, Returns, and Raises sections where applicable."
            ),
            "input": f"```python\n{code_without_doc}\n```",
            "output": docstring,
        })

    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    path = OUTPUT_ROOT / "docs_generator" / "train.jsonl"
    count = write_jsonl(records, path)
    logger.info("docs_generator: %d records -> %s", count, path)
    return count


def _strip_docstring(code: str) -> str:
    """Attempt to remove the docstring from a Python function for training input."""
    lines = code.split("\n")
    result = []
    in_docstring = False
    docstring_char = None
    found_docstring = False

    for line in lines:
        stripped = line.strip()
        if not found_docstring and not in_docstring:
            # Handle r""", u""", b""" prefixes and both quote styles
            check = stripped.lstrip("rRuUbB")
            if check.startswith('"""') or check.startswith("'''"):
                docstring_char = check[:3]
                in_docstring = True
                # Single-line docstring: """text""" on one line
                rest_after_open = check[3:]
                if docstring_char in rest_after_open:
                    in_docstring = False
                    found_docstring = True
                continue
            result.append(line)
        elif in_docstring:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                found_docstring = True
            continue
        else:
            result.append(line)

    return "\n".join(result)


# ============================================================================
# Builder registry
# ============================================================================

BUILDERS = {
    "code_writer": build_code_writer,
    "test_generator": build_test_generator,
    "static_reviewer": build_static_reviewer,
    "security_auditor": build_security_auditor,
    "performance_optimizer": build_performance_optimizer,
    "docs_generator": build_docs_generator,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build real Python training datasets from HuggingFace Hub"
    )
    parser.add_argument(
        "--agent",
        choices=list(BUILDERS.keys()),
        default=None,
        help="Build dataset for a specific agent (default: all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset to N samples (useful for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset builders and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available dataset builders:")
        for name, fn in BUILDERS.items():
            print(f"  {name:25s} — {fn.__doc__.strip().split(chr(10))[0]}")
        return

    random.seed(args.seed)

    agents_to_build = [args.agent] if args.agent else list(BUILDERS.keys())

    total = 0
    for agent_name in agents_to_build:
        logger.info("=" * 60)
        logger.info("Building: %s", agent_name)
        logger.info("=" * 60)
        try:
            count = BUILDERS[agent_name](max_samples=args.max_samples)
            total += count
            logger.info("Done: %s (%d records)\n", agent_name, count)
        except Exception as e:
            logger.error("FAILED: %s — %s", agent_name, e, exc_info=True)

    logger.info("=" * 60)
    logger.info("All done. Total records: %d", total)
    logger.info("Output directory: %s", OUTPUT_ROOT)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
