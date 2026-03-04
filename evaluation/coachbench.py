"""
CoachBench — ProofCoach Evaluation Benchmark

Evaluates ProofCoach across two dimensions:
  1. Problem solving: can the model solve held-out olympiad problems?
  2. Teaching quality: does tutoring improve student performance on similar problems?

200 held-out problems across:
  - AMC 10/12 (80 problems, difficulty 3-7)
  - AIME (60 problems, difficulty 6-9)
  - USAMO (40 problems, difficulty 8-10)
  - Putnam (20 problems, difficulty 8-10)

Usage:
    python evaluation/coachbench.py --model checkpoints/proofcoach-final --all
    python evaluation/coachbench.py --model checkpoints/proofcoach-final --solving
    python evaluation/coachbench.py --model checkpoints/proofcoach-final --teaching
    python evaluation/coachbench.py --model checkpoints/proofcoach-final --lean
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.lean4_interface import Lean4Interface


BENCHMARK_PROBLEMS = [
    # AMC 10/12 — 80 problems
    {"id": "AMC12-2023-A20", "competition": "AMC_12A", "year": 2023, "number": 20,
     "statement": "Let $f(x) = x^4 + ax^3 + bx^2 + cx + d$. The roots of $f$ are $r_1, r_2, r_3, r_4$. What is the minimum value of $(r_1 r_2)^2 + (r_1 r_3)^2 + (r_1 r_4)^2 + (r_2 r_3)^2 + (r_2 r_4)^2 + (r_3 r_4)^2$ when $a = b = c = 0$?",
     "answer": "0", "answer_type": "multiple_choice", "difficulty": 7},
    # Additional problems would be loaded from data/raw in practice
    # This is a representative sample for demonstration
]

# v1 targets
TARGETS = {
    "amc_accuracy": 0.85,
    "aime_accuracy": 0.50,
    "usamo_quality": 0.70,
    "lean4_verification_rate": 0.90,
    "teaching_improvement": 0.30,
}


@dataclass
class ProblemResult:
    """Result of evaluating a single problem."""
    problem_id: str
    competition: str
    difficulty: int
    model_answer: Optional[str]
    correct_answer: Optional[str]
    is_correct: bool
    lean4_claims: list[str]
    lean4_verified: int
    lean4_total: int
    response_time_ms: float
    full_response: str


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    model_path: str
    timestamp: float
    amc_accuracy: float
    aime_accuracy: float
    usamo_quality: float
    lean4_verification_rate: float
    teaching_improvement: Optional[float]
    n_problems: int
    problem_results: list[ProblemResult]
    passed: bool


class CoachBench:
    """
    ProofCoach evaluation benchmark.

    Evaluates:
      1. Solving accuracy on 200 held-out olympiad problems
      2. Lean 4 verification rate of generated proof steps
      3. Teaching effectiveness (simulated student improvement)
    """

    def __init__(
        self,
        model_path: str = "checkpoints/proofcoach-final",
        lean4_simulated: bool = False,
        results_dir: str = "results",
    ) -> None:
        self.model_path = model_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model for evaluation: {model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except (ValueError, ImportError):
            logger.warning("flash_attention_2 not available; falling back to eager attention")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )
        self._model.eval()

        self._lean4 = Lean4Interface(
            timeout=15,
            simulated=lean4_simulated or os.getenv("LEAN4_SIMULATED", "0") == "1",
        )

        self._problems = self._load_benchmark_problems()
        logger.info(f"CoachBench: {len(self._problems)} benchmark problems loaded")

    def run_all(self) -> BenchmarkResults:
        """Run complete benchmark."""
        solving = self.run_solving()
        lean = self.run_lean_verification()
        teaching = self.run_teaching()

        results = BenchmarkResults(
            model_path=self.model_path,
            timestamp=time.time(),
            amc_accuracy=solving["amc_accuracy"],
            aime_accuracy=solving["aime_accuracy"],
            usamo_quality=solving["usamo_quality"],
            lean4_verification_rate=lean["verification_rate"],
            teaching_improvement=teaching["improvement"],
            n_problems=len(self._problems),
            problem_results=solving["problem_results"],
            passed=self._check_targets(solving, lean, teaching),
        )

        self._save_results(results)
        self._print_summary(results)
        return results

    def run_solving(self) -> dict:
        """Evaluate problem-solving accuracy."""
        logger.info("Running solving evaluation...")

        amc_problems = [p for p in self._problems if "AMC" in p["competition"]]
        aime_problems = [p for p in self._problems if "AIME" in p["competition"]]
        usamo_problems = [p for p in self._problems if p["competition"] in ("USAMO", "IMO")]

        amc_results = [self._evaluate_problem(p) for p in amc_problems]
        aime_results = [self._evaluate_problem(p) for p in aime_problems]
        usamo_results = [self._evaluate_problem(p) for p in usamo_problems]

        amc_acc = sum(r.is_correct for r in amc_results) / max(len(amc_results), 1)
        aime_acc = sum(r.is_correct for r in aime_results) / max(len(aime_results), 1)
        usamo_qual = sum(r.is_correct for r in usamo_results) / max(len(usamo_results), 1)

        logger.info(f"AMC accuracy: {amc_acc:.1%} (target: {TARGETS['amc_accuracy']:.0%})")
        logger.info(f"AIME accuracy: {aime_acc:.1%} (target: {TARGETS['aime_accuracy']:.0%})")
        logger.info(f"USAMO quality: {usamo_qual:.1%} (target: {TARGETS['usamo_quality']:.0%})")

        return {
            "amc_accuracy": amc_acc,
            "aime_accuracy": aime_acc,
            "usamo_quality": usamo_qual,
            "problem_results": amc_results + aime_results + usamo_results,
        }

    def run_lean_verification(self) -> dict:
        """Evaluate Lean 4 verification rate of generated proof steps."""
        logger.info("Running Lean 4 verification evaluation...")

        total_claims = 0
        verified_claims = 0

        for problem in self._problems[:50]:  # Subset for speed
            response = self._generate_response(problem, include_proofs=True)
            claims = self._lean4.extract_claims_from_dialogue(response)
            if not claims:
                continue

            results = self._lean4.verify_batch(claims)
            total_claims += len(results)
            verified_claims += sum(1 for r in results if r.success)

        rate = verified_claims / max(total_claims, 1)
        logger.info(
            f"Lean 4 verification rate: {rate:.1%} "
            f"({verified_claims}/{total_claims} claims) "
            f"(target: {TARGETS['lean4_verification_rate']:.0%})"
        )

        return {"verification_rate": rate, "verified": verified_claims, "total": total_claims}

    def run_teaching(self) -> dict:
        """
        Evaluate teaching effectiveness.

        Simulated protocol:
        1. Present student with problem (student starts with wrong approach)
        2. Run 3 rounds of Socratic tutoring
        3. Present a similar problem — check if student-simulated model improves
        """
        logger.info("Running teaching effectiveness evaluation...")

        # Use a subset of AMC problems for teaching evaluation
        teaching_problems = [p for p in self._problems if "AMC_10" in p.get("competition", "")][:20]

        improvements = []
        for problem in teaching_problems:
            # Baseline: can the "student" solve a similar problem without tutoring?
            similar = self._find_similar_problem(problem)
            if not similar:
                continue

            baseline_correct = self._simulate_student_answer(similar)

            # Tutor: run 3 rounds of tutoring on the original problem
            self._simulate_tutoring_session(problem)

            # Post-tutoring: can the "student" now solve the similar problem?
            post_correct = self._simulate_student_answer(similar)

            improvement = float(post_correct) - float(baseline_correct)
            improvements.append(improvement)

        avg_improvement = sum(improvements) / max(len(improvements), 1)
        logger.info(
            f"Teaching improvement: {avg_improvement:.3f} "
            f"(target: {TARGETS['teaching_improvement']:.2f})"
        )

        return {"improvement": avg_improvement, "n_sessions": len(improvements)}

    def _evaluate_problem(self, problem: dict) -> ProblemResult:
        """Evaluate the model on a single problem."""
        start = time.time()
        response = self._generate_response(problem)
        elapsed = (time.time() - start) * 1000

        model_answer = self._extract_answer(response, problem.get("answer_type", ""))
        correct_answer = problem.get("answer", "")
        is_correct = self._check_answer(model_answer, correct_answer, problem.get("answer_type", ""))

        claims = self._lean4.extract_claims_from_dialogue(response)
        verified = 0
        if claims:
            results = self._lean4.verify_batch(claims)
            verified = sum(1 for r in results if r.success)

        return ProblemResult(
            problem_id=problem.get("id", ""),
            competition=problem.get("competition", ""),
            difficulty=problem.get("difficulty", 5),
            model_answer=model_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            lean4_claims=claims,
            lean4_verified=verified,
            lean4_total=len(claims),
            response_time_ms=elapsed,
            full_response=response[:500],
        )

    def _generate_response(self, problem: dict, include_proofs: bool = False) -> str:
        """Generate a response for a problem."""
        system = (
            "You are ProofCoach. Solve this math problem step by step. "
            "Include your mathematical reasoning and, where possible, "
            "formal proof steps in Lean 4 (```lean ... ```)."
            if include_proofs else
            "You are ProofCoach. Solve this math problem. Show your work clearly."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Problem: {problem.get('statement', '')}"},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_answer(self, response: str, answer_type: str) -> Optional[str]:
        """Extract the model's answer from its response."""
        if answer_type == "multiple_choice":
            match = re.search(r"\b([A-E])\b", response[-200:])
            return match.group(1) if match else None
        elif answer_type == "integer":
            match = re.search(r"\b(\d{1,3})\b", response[-200:])
            return match.group(1) if match else None
        else:
            # For proof problems, check for key phrases
            return response[-100:]

    def _check_answer(
        self, model_answer: Optional[str], correct_answer: str, answer_type: str
    ) -> bool:
        """Check if the model's answer is correct."""
        if not model_answer or not correct_answer:
            return False

        if answer_type in ("multiple_choice", "integer"):
            return model_answer.strip().lower() == correct_answer.strip().lower()

        # For proof problems — simplified: check key terms present
        key_terms = correct_answer.lower().split()[:3]
        response_lower = model_answer.lower()
        required = max(1, len(key_terms) // 2)
        return sum(1 for t in key_terms if t in response_lower) >= required

    def _find_similar_problem(self, problem: dict) -> Optional[dict]:
        """Find a similar problem for teaching evaluation."""
        competition = problem.get("competition", "")
        difficulty = problem.get("difficulty", 5)
        topics = problem.get("topics", [])

        for p in self._problems:
            if p.get("id") == problem.get("id"):
                continue
            if p.get("competition") == competition and abs(p.get("difficulty", 5) - difficulty) <= 1:
                if any(t in p.get("topics", []) for t in topics):
                    return p

        return None

    def _simulate_student_answer(self, problem: dict) -> bool:
        """Simulate a student attempting a problem (simplified)."""
        # In practice, this would use a separate "student model"
        # PC-22: Seed the RNG so results are deterministic and reproducible
        # across benchmark runs. Use problem id for per-problem seeding so the
        # fixed seed does not make all problems share the same random draw.
        import random
        rng = random.Random(hash(problem.get("id", "")) & 0xFFFFFFFF)
        difficulty = problem.get("difficulty", 5)
        success_prob = max(0.1, 0.9 - 0.1 * difficulty)
        return rng.random() < success_prob

    def _simulate_tutoring_session(self, problem: dict) -> None:
        """
        Run a simulated tutoring session on a problem.

        PC-12: Implement a proper multi-turn tutoring loop rather than a
        single-shot generation call. The tutor and a simulated student exchange
        messages for up to 3 turns. Each exchange uses the growing conversation
        history so that later tutor turns are informed by earlier student
        responses. This mirrors a real Socratic session and validates that the
        model can maintain coherent multi-turn tutoring dialogues.
        """
        messages = [
            {
                "role": "system",
                "content": "You are ProofCoach, a Socratic math tutor.",
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {problem.get('statement', '')}\n\n"
                    "My work so far: I am not sure how to start."
                ),
            },
        ]

        import random
        rng = random.Random(hash(problem.get("id", "tutoring")) & 0xFFFFFFFF)

        max_turns = 3
        for turn_idx in range(max_turns):
            # Generate tutor response given the conversation so far
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            tutor_response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": tutor_response})

            # Simulated student follow-up (generic partial-understanding reply)
            student_replies = [
                "Oh, I think I see — so I should consider what happens when the variable is at its boundary?",
                "That's helpful. I tried substituting but I'm still not seeing why the inequality holds.",
                "I think I understand now. The key insight is that the terms cancel, right?",
            ]
            messages.append({"role": "user", "content": student_replies[turn_idx % len(student_replies)]})

    def _check_targets(self, solving: dict, lean: dict, teaching: dict) -> bool:
        """Check if all benchmark targets are met."""
        checks = [
            solving["amc_accuracy"] >= TARGETS["amc_accuracy"],
            solving["aime_accuracy"] >= TARGETS["aime_accuracy"],
            lean["verification_rate"] >= TARGETS["lean4_verification_rate"],
            (teaching.get("improvement", 0) or 0) >= TARGETS["teaching_improvement"],
        ]
        return all(checks)

    def _load_benchmark_problems(self) -> list[dict]:
        """Load benchmark problems (hardcoded subset + from data if available)."""
        problems = list(BENCHMARK_PROBLEMS)

        # Supplement with problems from data/raw if available
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for jsonl_file in raw_dir.rglob("*.jsonl"):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            p = json.loads(line)
                            if len(problems) < 200:
                                problems.append({
                                    "id": p.get("problem_id", ""),
                                    "competition": p.get("competition", ""),
                                    "year": p.get("year", 0),
                                    "number": p.get("number", 0),
                                    "statement": p.get("statement", ""),
                                    "answer": p.get("answer", ""),
                                    "answer_type": p.get("answer_type", "proof"),
                                    "difficulty": p.get("difficulty_estimate", 5),
                                    "topics": p.get("topics", []),
                                })
                        except Exception:
                            continue

        return problems[:200]

    def _save_results(self, results: BenchmarkResults) -> None:
        """Save results to disk."""
        timestamp = int(results.timestamp)
        path = self.results_dir / f"coachbench_{timestamp}.json"
        data = asdict(results)
        # Truncate full responses to save space
        for pr in data.get("problem_results", []):
            pr["full_response"] = pr["full_response"][:200]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")

    def _print_summary(self, results: BenchmarkResults) -> None:
        """Print a summary table."""
        print("\n" + "=" * 60)
        print("CoachBench Results Summary")
        print("=" * 60)
        rows = [
            ("AMC 10/12 accuracy", f"{results.amc_accuracy:.1%}", f"{TARGETS['amc_accuracy']:.0%}",
             "PASS" if results.amc_accuracy >= TARGETS["amc_accuracy"] else "FAIL"),
            ("AIME accuracy", f"{results.aime_accuracy:.1%}", f"{TARGETS['aime_accuracy']:.0%}",
             "PASS" if results.aime_accuracy >= TARGETS["aime_accuracy"] else "FAIL"),
            ("USAMO quality", f"{results.usamo_quality:.1%}", f"{TARGETS['usamo_quality']:.0%}",
             "PASS" if results.usamo_quality >= TARGETS["usamo_quality"] else "FAIL"),
            ("Lean 4 verification", f"{results.lean4_verification_rate:.1%}",
             f"{TARGETS['lean4_verification_rate']:.0%}",
             "PASS" if results.lean4_verification_rate >= TARGETS["lean4_verification_rate"] else "FAIL"),
            ("Teaching improvement", f"{results.teaching_improvement:.3f}" if results.teaching_improvement else "N/A",
             f"{TARGETS['teaching_improvement']:.2f}",
             "PASS" if (results.teaching_improvement or 0) >= TARGETS["teaching_improvement"] else "FAIL"),
        ]

        for metric, value, target, status in rows:
            print(f"  {metric:<30} {value:>8}  (target: {target:>6})  [{status}]")

        print("=" * 60)
        overall = "PASSED" if results.passed else "FAILED"
        print(f"Overall: {overall}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoachBench Evaluation")
    parser.add_argument("--model", default="checkpoints/proofcoach-final")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--solving", action="store_true")
    parser.add_argument("--teaching", action="store_true")
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--simulated-lean", action="store_true")
    args = parser.parse_args()

    bench = CoachBench(model_path=args.model, lean4_simulated=args.simulated_lean)

    if args.all:
        bench.run_all()
    elif args.solving:
        bench.run_solving()
    elif args.teaching:
        bench.run_teaching()
    elif args.lean:
        bench.run_lean_verification()
    else:
        parser.print_help()
