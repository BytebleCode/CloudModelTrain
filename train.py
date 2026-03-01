#!/usr/bin/env python3
"""
CLI entrypoint for agent-specialized QLoRA fine-tuning.

Usage:
    python train.py --agent code_writer
    python train.py --agent security_auditor --run_name sec_v2
    python train.py --agent test_generator --resume outputs/test_generator/checkpoint-400
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a specialized coding-agent model (QLoRA on A100)",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent role name (e.g. code_writer, test_generator, security_auditor)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run name (defaults to timestamp)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override base model (e.g. codellama/CodeLlama-13b-hf)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override max sequence length",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable Flash Attention 2",
    )
    parser.add_argument(
        "--skip_pull",
        action="store_true",
        help="Skip dataset pulling (assume data is cached locally)",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only pull and validate dataset, don't train",
    )
    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """Convert CLI flags into a nested override dict."""
    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["name_or_path"] = args.model
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.epochs is not None:
        overrides.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.max_seq_length is not None:
        overrides.setdefault("training", {})["max_seq_length"] = args.max_seq_length
    if args.flash_attn:
        overrides.setdefault("flash_attention", {})["enabled"] = True
    return overrides


def main():
    args = parse_args()

    # --- Resolve config ---
    from src.config import resolve_config, save_config_snapshot

    cli_overrides = build_cli_overrides(args)
    cfg = resolve_config(args.agent, args.run_name, cli_overrides)

    logger.info("Agent: %s", args.agent)
    logger.info("Run name: %s", cfg["run_name"])
    logger.info("Output dir: %s", cfg["output_dir"])

    # --- Save config snapshot ---
    snapshot_path = save_config_snapshot(cfg, cfg["output_dir"])
    logger.info("Config snapshot saved: %s", snapshot_path)

    # --- Pull dataset ---
    agent_reg = cfg["agent_registry"]
    data_cache_dir = cfg["data_cache_dir"]
    dataset_uri = agent_reg["dataset_uri"]
    eval_uri = agent_reg.get("eval_uri", "")

    if not args.skip_pull:
        from src.data.pull import pull_dataset

        logger.info("Pulling dataset: %s", dataset_uri)
        train_data_path = pull_dataset(dataset_uri, data_cache_dir)
        logger.info("Train data ready: %s", train_data_path)

        eval_data_path = None
        if eval_uri:
            eval_data_path = pull_dataset(eval_uri, data_cache_dir)
            logger.info("Eval data ready: %s", eval_data_path)
    else:
        # Expect data already in cache
        train_data_path = Path(data_cache_dir)
        jsonl_files = list(train_data_path.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("No JSONL files found in %s (use --skip_pull only if data is cached)", data_cache_dir)
            sys.exit(1)
        train_data_path = jsonl_files[0]
        eval_data_path = None

    # --- Validate ---
    from src.data.validate import validate_jsonl

    fmt = agent_reg["format"]
    fields = agent_reg.get("fields", {})
    if fmt == "instruction":
        required = list(fields.values())
    elif fmt == "chat":
        required = [fields.get("messages", "messages")]
    else:
        required = None

    sha = agent_reg.get("sha256", "") or None
    record_count = validate_jsonl(train_data_path, required_fields=required, expected_sha256=sha)
    logger.info("Validated %d records", record_count)

    if args.validate_only:
        logger.info("Validation complete — exiting (--validate_only).")
        return

    # --- Prepare dataset ---
    from transformers import AutoTokenizer
    from src.data.prepare import prepare_dataset

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = prepare_dataset(
        data_path=train_data_path,
        eval_path=Path(eval_data_path) if eval_data_path else None,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    # --- Train ---
    from src.train.run import run_training

    output_path = run_training(cfg, dataset, resume_from=args.resume)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Adapter:   %s/adapter", output_path)
    logger.info("Tokenizer: %s/tokenizer", output_path)
    logger.info("Logs:      %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
