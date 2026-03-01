"""
Dataset preparation: load JSONL, apply prompt templates, tokenize, optionally pack.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "instruction": (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ),
    "chat": None,  # handled separately via apply_chat_template
}


def _format_instruction(record: dict, fields: dict[str, str]) -> str:
    """Apply the instruction template, mapping JSONL fields to placeholders."""
    instruction = record.get(fields.get("instruction", "instruction"), "")
    inp = record.get(fields.get("input", "input"), "")
    output = record.get(fields.get("output", "output"), "")
    return TEMPLATES["instruction"].format(
        instruction=instruction, input=inp, output=output
    )


def _format_chat(record: dict, fields: dict[str, str], tokenizer) -> str:
    """Apply chat template using the tokenizer's built-in chat formatting."""
    messages_key = fields.get("messages", "messages")
    messages = record.get(messages_key, [])
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback: simple concatenation
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dataset(
    data_path: Path,
    eval_path: Path | None,
    tokenizer,
    cfg: dict,
) -> DatasetDict:
    """
    Load, format, tokenize, and optionally pack a dataset.

    Returns a DatasetDict with 'train' and 'eval' splits.
    """
    agent_reg = cfg["agent_registry"]
    fmt = agent_reg["format"]
    fields = agent_reg.get("fields", {})
    max_seq_length = cfg["training"]["max_seq_length"]
    packing = cfg.get("packing", {}).get("enabled", False)

    # --- Load raw records ---
    records = load_jsonl(data_path)
    logger.info("Loaded %d training records from %s", len(records), data_path)

    # --- Format into text ---
    def format_record(record: dict) -> str:
        if fmt == "instruction":
            return _format_instruction(record, fields)
        elif fmt == "chat":
            return _format_chat(record, fields, tokenizer)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    texts = [format_record(r) for r in records]

    # --- Build HF Dataset ---
    ds = Dataset.from_dict({"text": texts})

    # --- Train / eval split ---
    if eval_path and eval_path.exists():
        eval_records = load_jsonl(eval_path)
        eval_texts = [format_record(r) for r in eval_records]
        eval_ds = Dataset.from_dict({"text": eval_texts})
        ds_dict = DatasetDict({"train": ds, "eval": eval_ds})
    else:
        eval_size = cfg.get("data", {}).get("eval_size", 0.05)
        seed = cfg.get("data", {}).get("seed", 42)
        split = ds.train_test_split(test_size=eval_size, seed=seed)
        ds_dict = DatasetDict({"train": split["train"], "eval": split["test"]})

    logger.info(
        "Dataset splits — train: %d, eval: %d",
        len(ds_dict["train"]),
        len(ds_dict["eval"]),
    )

    # --- Tokenize ---
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds_dict = ds_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # --- Packing (optional) ---
    if packing:
        logger.info("Packing sequences to max_seq_length=%d", max_seq_length)
        eos_id = tokenizer.eos_token_id or 0
        ds_dict["train"] = _pack_dataset(
            ds_dict["train"], max_seq_length, eos_id
        )

    return ds_dict


def _pack_dataset(
    dataset: Dataset,
    max_seq_length: int,
    eos_token_id: int,
) -> Dataset:
    """
    Concatenate tokenized samples with EOS separators, then chunk into
    fixed-length sequences. Labels are set to -100 at sample boundaries
    (the EOS separator token) so the model doesn't learn to predict
    across unrelated samples.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    buffer_ids: list[int] = []
    buffer_mask: list[int] = []
    buffer_labels: list[int] = []

    for row in dataset:
        # Insert EOS separator between samples
        if buffer_ids:
            buffer_ids.append(eos_token_id)
            buffer_mask.append(1)
            buffer_labels.append(-100)  # mask boundary token

        buffer_ids.extend(row["input_ids"])
        buffer_mask.extend(row["attention_mask"])
        buffer_labels.extend(row["labels"])

        while len(buffer_ids) >= max_seq_length:
            all_input_ids.append(buffer_ids[:max_seq_length])
            all_attention_mask.append(buffer_mask[:max_seq_length])
            all_labels.append(buffer_labels[:max_seq_length])
            buffer_ids = buffer_ids[max_seq_length:]
            buffer_mask = buffer_mask[max_seq_length:]
            buffer_labels = buffer_labels[max_seq_length:]

    # Drop the remainder (shorter than max_seq_length) to keep shapes uniform.
    logger.info(
        "Packing: %d samples -> %d packed sequences (dropped %d trailing tokens)",
        len(dataset),
        len(all_input_ids),
        len(buffer_ids),
    )

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    )
