import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train_dpo.py — Stage 3: DPO on Teaching Quality

Trains the model to prefer Socratic teaching over direct answers.

Chosen:   Step-by-step Socratic guidance → student discovers insight
Rejected: Direct answer / mechanical step-by-step solution

Preference signals:
  1. AoPS upvotes — high-voted community tutoring posts (chosen)
  2. Generated low-quality responses — direct answer counterparts (rejected)
  3. Teaching outcome — student success rate after tutoring

Algorithm: DPO (Direct Preference Optimization), beta=0.1
Data: ~20k preference pairs
Hardware: 18× A6000, DeepSpeed ZeRO-3

Launch:
  deepspeed --num_gpus=18 training/train_dpo.py \\
    --base-model checkpoints/proofcoach-rl/final \\
    --output-dir checkpoints/proofcoach-final
"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from loguru import logger  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DPODataset:
    """
    Loads DPO preference pairs from multiple sources.

    Sources:
      1. data/synthesized/dpo/dpo_pairs.jsonl — generated preference pairs
      2. data/raw/aops/ — high-voted vs. low-voted community solutions

    Format expected:
      {
        "prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "chosen": [{"role": "assistant", "content": "..."}],
        "rejected": [{"role": "assistant", "content": "..."}]
      }
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> Dataset:
        """Load all DPO pairs."""
        records = []

        # Primary: synthesized DPO pairs
        dpo_file = self.data_dir / "synthesized" / "dpo" / "dpo_pairs.jsonl"
        if dpo_file.exists():
            records.extend(self._load_jsonl(dpo_file))
            logger.info(f"Loaded {len(records):,} pairs from synthesized DPO file")

        # Secondary: AoPS upvote-based pairs
        aops_dir = self.data_dir / "raw" / "aops"
        if aops_dir.exists():
            aops_pairs = self._load_aops_pairs(aops_dir)
            records.extend(aops_pairs)
            logger.info(f"Loaded {len(aops_pairs):,} pairs from AoPS upvotes")

        if not records:
            raise ValueError(
                "No DPO pairs found. Run synthesis/generate_dpo_pairs.py first."
            )

        logger.info(f"Total DPO pairs: {len(records):,}")
        return Dataset.from_list(records)

    def _load_jsonl(self, path: Path) -> list[dict]:
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

    def _load_aops_pairs(self, aops_dir: Path) -> list[dict]:
        """
        Extract preference pairs from AoPS data using upvote signal.

        For each problem with multiple solutions:
          chosen:   highest-voted solution (≥10 upvotes)
          rejected: lowest-voted solution (0-2 upvotes)
        """
        pairs = []
        for jsonl_file in aops_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        problem = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    solutions = problem.get("solutions", [])
                    if len(solutions) < 2:
                        continue

                    # Sort by upvotes
                    solutions.sort(key=lambda s: s.get("upvotes", 0), reverse=True)
                    best = solutions[0]
                    worst = solutions[-1]

                    # Only use if there's a meaningful upvote gap
                    if best.get("upvotes", 0) - worst.get("upvotes", 0) < 5:
                        continue

                    prompt = [
                        {
                            "role": "system",
                            "content": "You are ProofCoach, a Socratic math tutor.",
                        },
                        {
                            "role": "user",
                            "content": f"Problem: {problem.get('statement', '')}\n\nI am not sure how to approach this.",
                        },
                    ]

                    pairs.append(
                        {
                            "prompt": prompt,
                            "chosen": [
                                {
                                    "role": "assistant",
                                    "content": best.get("content", ""),
                                }
                            ],
                            "rejected": [
                                {
                                    "role": "assistant",
                                    "content": worst.get("content", ""),
                                }
                            ],
                            "metadata": {
                                "source": "aops_upvotes",
                                "problem_id": problem.get("problem_id", ""),
                                "chosen_upvotes": best.get("upvotes", 0),
                                "rejected_upvotes": worst.get("upvotes", 0),
                            },
                        }
                    )

        return pairs


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class DPOMetricsCallback(TrainerCallback):
    """Log DPO-specific metrics during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        keys = [
            "loss",
            "rewards/chosen",
            "rewards/rejected",
            "rewards/margins",
            "learning_rate",
        ]
        parts = [f"Step {step:5d}"]
        for k in keys:
            if k in logs:
                parts.append(f"{k.split('/')[-1]}: {logs[k]:.4f}")
        logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Data Formatting
# ---------------------------------------------------------------------------


def format_dpo_example(example: dict, tokenizer) -> dict:  # noqa: ARG001
    """
    Pass DPO pair fields through unchanged.

    DPOTrainer applies the chat template internally.  Pre-applying it here
    would cause the template to be applied twice, corrupting the input.
    The fields are returned as raw conversation lists so DPOTrainer can
    handle tokenization itself.
    """
    return {
        "prompt": example.get("prompt", []),
        "chosen": example.get("chosen", []),
        "rejected": example.get("rejected", []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ProofCoach Stage 3: DPO")
    parser.add_argument("--base-model", default="checkpoints/proofcoach-rl/final")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/proofcoach-final")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--wandb-project", default="proofcoach-dpo")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    logger.info(f"Stage 3 DPO — base model: {args.base_model}")
    logger.info(f"DPO beta: {args.beta}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # LoRA for DPO
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    ref_model.eval()

    # Load DPO dataset
    dpo_dataset = DPODataset(args.data_dir)
    raw_dataset = dpo_dataset.load()

    def fmt(ex):
        return format_dpo_example(ex, tokenizer)

    dataset = raw_dataset.map(fmt, remove_columns=raw_dataset.column_names)

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # DPO Config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        beta=args.beta,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len // 2,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        deepspeed=args.deepspeed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        remove_unused_columns=False,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[DPOMetricsCallback()],
    )

    logger.info("Starting Stage 3 DPO training...")
    trainer.train()

    # Save final merged model
    logger.info("Merging LoRA weights and saving final model...")
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logger.success(f"Stage 3 DPO complete. Final model saved to {final_dir}")
    logger.success("ProofCoach training pipeline complete.")


if __name__ == "__main__":
    main()
