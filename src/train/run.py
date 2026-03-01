"""
Core training logic: QLoRA fine-tuning with PEFT on A100/H200.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.train.callbacks import CostEstimatorCallback, EvalLossDivergenceCallback

logger = logging.getLogger(__name__)


def _build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    qcfg = cfg["quantization"]
    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )


def _build_lora_config(cfg: dict) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )


def _build_training_args(cfg: dict, output_dir: str, has_eval: bool = True) -> TrainingArguments:
    tcfg = cfg["training"]
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        tf32=tcfg["tf32"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=tcfg["optim"],
        max_grad_norm=tcfg["max_grad_norm"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        dataloader_pin_memory=tcfg["dataloader_pin_memory"],
        report_to=tcfg["report_to"],
        seed=cfg.get("data", {}).get("seed", 42),
    )

    if has_eval:
        kwargs["eval_steps"] = tcfg["eval_steps"]
        kwargs["eval_strategy"] = "steps"
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False

    return TrainingArguments(**kwargs)


def run_training(
    cfg: dict,
    dataset: DatasetDict,
    tokenizer,
    resume_from: str | None = None,
) -> Path:
    """
    Execute QLoRA fine-tuning.

    Args:
        cfg: Resolved config dict.
        dataset: DatasetDict with 'train' and 'eval' splits.
        tokenizer: Pre-loaded tokenizer (avoids double-load issues with Devstral/Tekken).
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the output directory containing the final adapter.
    """
    output_dir = cfg["output_dir"]
    model_name = cfg["model"]["name_or_path"]
    flash_attn = cfg.get("flash_attention", {}).get("enabled", False)

    # --- Enable TF32 globally on A100/H200 ---
    if cfg["training"].get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Quantized model ---
    logger.info("Loading model with 4-bit quantization: %s", model_name)
    bnb_config = _build_bnb_config(cfg)

    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    )
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    )

    # --- LoRA ---
    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- Training arguments ---
    has_eval = "eval" in dataset and len(dataset["eval"]) > 0
    training_args = _build_training_args(cfg, output_dir, has_eval=has_eval)

    # --- Callbacks ---
    callbacks = [
        CostEstimatorCallback(
            gpu_hour_price=cfg.get("cost", {}).get("gpu_hour_price", 1.10)
        ),
    ]
    if has_eval:
        callbacks.append(EvalLossDivergenceCallback(threshold=1.5, patience=3))

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("Starting training — output: %s", output_dir)
    trainer.train(resume_from_checkpoint=resume_from)

    # --- Save final adapter + tokenizer ---
    adapter_dir = Path(output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))

    tokenizer_dir = Path(output_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))

    logger.info("Adapter saved to: %s", adapter_dir)
    logger.info("Tokenizer saved to: %s", tokenizer_dir)

    return Path(output_dir)
