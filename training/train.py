import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train.py — Stage 1: SFT on Socratic tutoring dialogues

Fine-tunes Qwen2.5-7B-Coder-Instruct on 180k+ tutoring interaction pairs.

Hardware target: 18× A6000 (48GB each)
Strategy:
  - LoRA rank 64 (adapts q, k, v, o, gate, up, down projections)
  - DeepSpeed ZeRO-3 with CPU offload
  - 3 epochs, ~6h on 18× A6000

Launch:
  deepspeed --num_gpus=18 training/train.py \\
    --deepspeed training/configs/deepspeed_zero3.json \\
    --model Qwen/Qwen2.5-7B-Coder-Instruct \\
    --data-dir data/training \\
    --output-dir checkpoints/proofcoach-sft
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


def load_dataset(data_dir: str) -> tuple[Dataset, Dataset]:
    """
    Load training data from data/training/*.jsonl files.

    Supports both a single train/val split and a glob of multiple JSONL files.
    If train.jsonl and val.jsonl exist, use them directly.
    Otherwise, load all *.jsonl files and split 95/5.
    """
    data_path = Path(data_dir)

    def read_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    train_file = data_path / "train.jsonl"
    val_file = data_path / "val.jsonl"

    if train_file.exists() and val_file.exists():
        train_records = read_jsonl(train_file)
        val_records = read_jsonl(val_file)
    else:
        # Load all *.jsonl files and split
        all_records = []
        for jsonl_path in sorted(data_path.glob("*.jsonl")):
            all_records.extend(read_jsonl(jsonl_path))

        if not all_records:
            raise FileNotFoundError(
                f"No training data found in {data_dir}. "
                "Run synthesis pipeline first."
            )

        split_idx = int(len(all_records) * 0.95)
        train_records = all_records[:split_idx]
        val_records = all_records[split_idx:]

    logger.info(f"Train: {len(train_records):,} examples, Val: {len(val_records):,} examples")
    return Dataset.from_list(train_records), Dataset.from_list(val_records)


def format_to_text(example: dict, tokenizer) -> dict:
    """
    Convert conversation format to a single text string.

    Supports both ShareGPT format (from/value) and OpenAI format (role/content).
    """
    conversations = example.get("conversations", [])

    # Handle both ShareGPT and OpenAI conversation formats
    messages = []
    system_msgs = []
    for turn in conversations:
        role = turn.get("role") or turn.get("from", "")
        content = turn.get("content") or turn.get("value", "")

        if role == "system":
            system_msgs.append({"role": "system", "content": content})
        elif role in ("user", "human"):
            messages.append({"role": "user", "content": content})
        elif role in ("assistant", "gpt"):
            messages.append({"role": "assistant", "content": content})

    if system_msgs:
        messages = system_msgs + messages

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def build_lora_config() -> LoraConfig:
    """Build LoRA configuration for SFT."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


class PrintMetricsCallback(TrainerCallback):
    """Log training metrics in a clean format."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", "")
        lr = logs.get("learning_rate", "")
        epoch = logs.get("epoch", "")

        if loss:
            logger.info(
                f"Step {step:5d} | epoch {epoch:.2f} | loss {loss:.4f} | lr {lr:.2e}"
            )


def main():
    parser = argparse.ArgumentParser(description="ProofCoach Stage 1: SFT")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Coder-Instruct")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--output-dir", default="checkpoints/proofcoach-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--wandb-project", default="proofcoach-sft")
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both --fp16 and --bf16. Specify only one.")

    os.environ["WANDB_PROJECT"] = args.wandb_project

    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_dataset, val_dataset = load_dataset(args.data_dir)

    # Format to text
    def fmt(ex):
        return format_to_text(ex, tokenizer)

    train_dataset = train_dataset.map(fmt, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(fmt, remove_columns=val_dataset.column_names)

    # Training arguments — evaluation strategy depends on whether a validation
    # set was actually loaded. If not, disable eval to avoid load_best_model_at_end
    # silently using the final checkpoint instead of the true best.
    has_val = len(val_dataset) > 0

    # Only report to wandb when a key is configured; otherwise fall back to
    # "none" so the run doesn't fail if WANDB_API_KEY is not set.
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=10,
        eval_strategy="steps" if has_val else "no",
        eval_steps=500 if has_val else None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        deepspeed=args.deepspeed,
        report_to=report_to,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[PrintMetricsCallback()],
    )

    logger.info("Starting Stage 1 SFT training...")
    trainer.train()

    # Save final model (merge LoRA into base weights)
    logger.info("Merging LoRA weights and saving...")
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logger.success(f"Stage 1 SFT complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
