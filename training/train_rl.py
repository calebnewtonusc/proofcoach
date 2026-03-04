import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train_rl.py — Stage 2: GRPO RL with Lean 4 Verification Reward

The core innovation: the reward signal is Lean 4 proof verification.
When the model generates a tutoring step, the mathematical claims
are extracted and formally type-checked by Lean 4.

Reward function:
  +1.0  — Lean 4 verifies the proof step (claim is mathematically correct)
  +0.3  — Correct numerical answer without formal proof
   0.0  — Lean 4 timeout
  -1.0  — Lean 4 rejects (claim is mathematically incorrect)

Algorithm: GRPO (Group Relative Policy Optimization)
  - N=4 completions per prompt
  - Group-relative advantages (normalize within group)
  - KL penalty (beta=0.02) vs. frozen SFT reference

Hardware: 18× A6000, DeepSpeed ZeRO-2 (no CPU offload for speed)

Launch:
  deepspeed --num_gpus=18 training/train_rl.py \\
    --base-model checkpoints/proofcoach-sft/final \\
    --output-dir checkpoints/proofcoach-rl
"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from loguru import logger  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from core.lean4_interface import Lean4Interface  # noqa: E402


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------


class Lean4VerificationReward:
    """
    Reward function based on Lean 4 proof verification.

    For each generated tutoring step:
    1. Extract mathematical claims (```lean ... ``` blocks or heuristics)
    2. Submit each claim to Lean 4 type-checker
    3. Compute reward based on verification outcome
    """

    def __init__(
        self,
        timeout: int = 10,
        simulated: bool = False,
    ) -> None:
        self._lean4 = Lean4Interface(
            timeout=timeout,
            simulated=simulated or os.getenv("LEAN4_SIMULATED", "0") == "1",
        )
        logger.info(
            f"Lean4VerificationReward initialized (simulated={self._lean4.simulated})"
        )

    def compute_rewards(
        self,
        prompts: list[str],
        completions: list[str],
        problem_answers: Optional[list[Optional[str]]] = None,
    ) -> list[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            prompts: Input prompts (not used for reward, but kept for API consistency)
            completions: Generated tutoring responses
            problem_answers: Correct answers for partial credit scoring

        Returns:
            List of reward values in [-1.0, +1.0]
        """
        rewards = []
        for i, completion in enumerate(completions):
            answer = problem_answers[i] if problem_answers else None
            reward = self._reward_single(completion, answer)
            rewards.append(reward)
        return rewards

    def _reward_single(
        self,
        completion: str,
        correct_answer: Optional[str] = None,
    ) -> float:
        """Compute reward for a single completion."""
        # Extract Lean 4 claims
        claims = self._lean4.extract_claims_from_dialogue(completion)

        if not claims:
            # No formal claims — check for correct numerical answer
            if correct_answer and self._contains_correct_answer(
                completion, correct_answer
            ):
                return 0.3  # Partial credit
            return (
                -0.2
            )  # Slight negative for no formal claims (but not harshly penalized)

        # Verify all claims
        results = self._lean4.verify_batch(claims)

        if not results:
            return 0.0

        # Aggregate reward across claims
        verified = sum(1 for r in results if r.success)
        total = len(results)
        timed_out = sum(1 for r in results if r.reward == 0.0)

        if total == verified:
            return 1.0  # All claims verified
        elif verified > 0:
            # Partial verification
            verified_ratio = verified / (total - timed_out + 1e-6)
            return 0.5 * verified_ratio
        else:
            return -1.0  # All claims failed

    def _contains_correct_answer(self, completion: str, correct_answer: str) -> bool:
        """Check if the completion contains the correct answer."""
        import re

        # Normalize: remove whitespace and compare
        answer_norm = re.sub(r"\s+", "", correct_answer.lower())
        completion_lower = completion.lower()

        # Look for the answer pattern
        patterns = [
            rf"\b{re.escape(answer_norm)}\b",
            rf"answer.*?{re.escape(answer_norm)}",
            rf"={re.escape(answer_norm)}",
        ]
        return any(re.search(p, completion_lower) for p in patterns)


# ---------------------------------------------------------------------------
# GRPO Algorithm
# ---------------------------------------------------------------------------


def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Compute group-relative advantages for GRPO.

    For a group of N completions for the same prompt:
      advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
    """
    import numpy as np

    rewards_np = np.array(rewards, dtype=np.float32)
    mean = rewards_np.mean()
    std = rewards_np.std() + 1e-8
    return ((rewards_np - mean) / std).tolist()


def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Compute log probabilities of response tokens given the prompt.

    Only computes log probs for the response portion (after response_start).
    """
    # Guard: if response_start is at or beyond the truncation boundary the
    # response slice will be empty — return zero log-prob with a warning.
    seq_len = input_ids.shape[1]
    if response_start >= seq_len - 1:
        logger.warning(
            f"response_start ({response_start}) >= seq_len-1 ({seq_len - 1}); "
            "completion was truncated. Returning zero log-prob."
        )
        return torch.zeros(
            input_ids.shape[0], device=input_ids.device, requires_grad=True
        )

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Shift logits/labels for causal LM
    logits = logits[:, response_start:-1, :]
    labels = input_ids[:, response_start + 1 :]

    # Guard: labels should be non-empty after slicing
    if labels.shape[1] == 0:
        logger.warning(
            "labels tensor has length 0 — completion was truncated. Returning zero log-prob."
        )
        return torch.zeros(
            input_ids.shape[0], device=input_ids.device, requires_grad=True
        )

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding
    mask = attention_mask[:, response_start + 1 :]
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=-1)  # (batch,)


def compute_grpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int,
    advantages: torch.Tensor,
    kl_coef: float = 0.02,
) -> torch.Tensor:
    """
    Compute the GRPO loss for a batch of completions.

    Loss = -mean(advantage * log_prob) + kl_coef * KL(policy || reference)
    """
    # Policy log probs
    policy_log_probs = compute_log_probs(
        model, input_ids, attention_mask, response_start
    )

    # Reference log probs (frozen SFT model)
    with torch.no_grad():
        ref_log_probs = compute_log_probs(
            ref_model, input_ids, attention_mask, response_start
        )

    # KL divergence penalty
    kl = policy_log_probs - ref_log_probs

    # GRPO loss
    loss = -(advantages * policy_log_probs).mean() + kl_coef * kl.mean()
    return loss


def generate_completions(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    n_completions: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
) -> list[list[str]]:
    """
    Generate N completions per prompt for GRPO.

    Returns: list of lists — outer indexed by prompt, inner by completion.
    """
    all_completions = []

    for prompt in prompts:
        completions = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        for _ in range(n_completions):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text)

        all_completions.append(completions)

    return all_completions


def pad_batch(sequences: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length."""
    max_len = max(s.shape[0] for s in sequences)
    padded = []
    for s in sequences:
        pad_len = max_len - s.shape[0]
        if pad_len > 0:
            s = F.pad(s, (0, pad_len), value=pad_value)
        padded.append(s)
    return torch.stack(padded)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """GRPO trainer for Lean 4 reward-based RL."""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        reward_fn: Lean4VerificationReward,
        optimizer: torch.optim.Optimizer,
        n_completions: int = 4,
        kl_coef: float = 0.02,
        max_new_tokens: int = 512,
        gradient_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.n_completions = n_completions
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.gradient_clip = gradient_clip

    def train_step(
        self,
        prompts: list[str],
        problem_answers: Optional[list[Optional[str]]] = None,
    ) -> dict:
        """
        Execute one GRPO training step.

        1. Generate N completions per prompt
        2. Score with Lean 4 reward
        3. Compute group-relative advantages
        4. Compute GRPO loss
        5. Backprop and update
        """
        # Step 1: Generate completions
        all_completions = generate_completions(
            self.model,
            self.tokenizer,
            prompts,
            n_completions=self.n_completions,
            max_new_tokens=self.max_new_tokens,
        )

        total_loss = 0.0
        total_reward = 0.0
        n_batches = 0

        for prompt_idx, (prompt, completions) in enumerate(
            zip(prompts, all_completions)
        ):
            # Step 2: Compute rewards
            answers = (
                [problem_answers[prompt_idx]] * self.n_completions
                if problem_answers
                else None
            )
            rewards = self.reward_fn.compute_rewards(
                [prompt] * self.n_completions,
                completions,
                problem_answers=answers,
            )

            # Step 3: Compute advantages
            advantages = compute_advantages(rewards)
            advantages_tensor = torch.tensor(advantages, device=self.model.device)
            total_reward += sum(rewards) / len(rewards)

            # Step 4: Compute GRPO loss over completions
            for comp_idx, (completion, adv) in enumerate(
                zip(completions, advantages_tensor)
            ):
                full_text = prompt + completion
                encoding = self.tokenizer(
                    full_text, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.model.device)
                prompt_enc = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )
                response_start = prompt_enc["input_ids"].shape[1]

                loss = compute_grpo_loss(
                    self.model,
                    self.ref_model,
                    encoding["input_ids"],
                    encoding["attention_mask"],
                    response_start,
                    adv.unsqueeze(0),
                    kl_coef=self.kl_coef,
                )
                # PC-13: Divide by n_completions before backward so that the
                # gradient is the mean over the completion group, not the sum.
                # Without this normalisation the effective learning rate would
                # scale with n_completions, making it harder to tune --lr
                # independently of --n-completions.
                (loss / self.n_completions).backward()
                total_loss += loss.item()
                n_batches += 1

        # Step 5: Update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": total_loss / max(n_batches, 1),
            "mean_reward": total_reward / len(prompts),
        }


def load_rl_prompts(data_dir: str) -> list[dict]:
    """Load training prompts for RL stage."""
    prompts = []
    train_file = Path(data_dir) / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            conversations = example.get("conversations", [])

            # Build the prompt (everything except last assistant turn)
            messages = []
            for turn in conversations[:-1]:
                role = turn.get("role") or turn.get("from", "")
                content = turn.get("content") or turn.get("value", "")
                if role == "system":
                    messages.append({"role": "system", "content": content})
                elif role in ("user", "human"):
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "gpt"):
                    messages.append({"role": "assistant", "content": content})

            if messages:
                prompts.append(
                    {
                        "messages": messages,
                        "answer": example.get("metadata", {}).get("answer"),
                    }
                )

    logger.info(f"Loaded {len(prompts):,} RL training prompts")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="ProofCoach Stage 2: GRPO RL")
    parser.add_argument("--base-model", default="checkpoints/proofcoach-sft/final")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--output-dir", default="checkpoints/proofcoach-rl")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)  # prompts per step
    parser.add_argument("--n-completions", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    # Note: DeepSpeed parallelism is handled by the `deepspeed --num_gpus=N`
    # launcher that wraps this script. The custom GRPOTrainer does not use
    # HF Trainer, so there is no Trainer.deepspeed config path. Do NOT pass
    # --deepspeed here; use the launcher flag instead:
    #   deepspeed --num_gpus=18 training/train_rl.py <args>
    parser.add_argument("--lean4-timeout", type=int, default=10)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()

    logger.info(f"Stage 2 GRPO RL — base model: {args.base_model}")
    logger.info(f"Lean 4 simulated: {args.simulated}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (trainable)
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
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
    model.train()

    # Frozen reference model
    ref_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Reward function
    reward_fn = Lean4VerificationReward(
        timeout=args.lean4_timeout,
        simulated=args.simulated,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        optimizer=optimizer,
        n_completions=args.n_completions,
        kl_coef=args.kl_coef,
    )

    # Load prompts
    prompts_data = load_rl_prompts(args.data_dir)
    random.shuffle(prompts_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting GRPO training: {args.steps} steps, batch_size={args.batch_size}"
    )

    for step in range(args.steps):
        # Sample batch
        batch_data = random.sample(
            prompts_data, min(args.batch_size, len(prompts_data))
        )

        prompts = []
        answers = []
        for d in batch_data:
            text = tokenizer.apply_chat_template(
                d["messages"], tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            answers.append(d.get("answer"))

        # Train step
        metrics = trainer.train_step(prompts, problem_answers=answers)

        if step % 10 == 0:
            logger.info(
                f"Step {step:4d}/{args.steps} | "
                f"loss {metrics['loss']:.4f} | "
                f"reward {metrics['mean_reward']:.3f}"
            )

        # Checkpoint every 500 steps — save LoRA adapter weights only (do NOT
        # call merge_and_unload here: that permanently destroys the PEFT structure
        # and makes subsequent training steps operate on a non-PEFT model).
        if step % 500 == 0 and step > 0:
            ckpt_dir = output_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info(f"Checkpoint saved: {ckpt_dir}")

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.success(f"Stage 2 GRPO RL complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
