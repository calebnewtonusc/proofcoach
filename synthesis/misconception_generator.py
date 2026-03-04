"""
Misconception Generator — Generates wrong approach → diagnosis → correction pairs

For each problem, generates:
  - A plausible student misconception (wrong approach)
  - Precise diagnosis of the error
  - Socratic corrective question

This makes misconception detection a first-class training objective,
not an afterthought.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for misconception_generator. "
        "Install with: pip install aiofiles"
    ) from exc
import aiohttp
import anthropic
from loguru import logger

from synthesis.prompts import MISCONCEPTION_SYSTEM, MISCONCEPTION_USER


MISCONCEPTION_CATEGORIES = [
    "assumed_continuity",
    "divided_by_zero",
    "theorem_out_of_domain",
    "if_vs_iff_confusion",
    "missed_edge_case",
    "double_counting",
    "modular_arithmetic_error",
    "necessary_vs_sufficient",
    "sign_error",
    "off_by_one",
    "assumed_independence",
    "incorrect_generalization",
]


@dataclass
class MisconceptionPair:
    """A student misconception with diagnosis and corrective question."""
    problem_id: str
    problem_statement: str
    correct_answer: Optional[str]
    student_wrong_approach: str
    student_wrong_answer: str
    misconception_type: str
    misconception_description: str
    corrective_question: str
    why_this_question: str
    quality_score: float


class MisconceptionGenerator:
    """
    Generates (wrong approach, diagnosis, corrective question) triples
    for use as misconception detection training data.

    Target: ~10,000 misconception pairs covering all competition types
    and all misconception categories.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 20,
        pairs_per_problem: int = 2,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self.pairs_per_problem = pairs_per_problem

        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_index = 0

        if backend == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running the misconception generator."
                )
            self._claude = anthropic.AsyncAnthropic(api_key=api_key)

        self._stats = {"generated": 0, "failed": 0}
        # PC-10: Shared aiohttp session for vLLM calls to enable connection pooling.
        self._vllm_session: Optional[aiohttp.ClientSession] = None

    async def generate_all(self) -> None:
        """Generate misconception pairs for all problems."""
        problems = self._load_problems()
        logger.info(f"Generating misconception pairs for {len(problems):,} problems...")

        output_file = self.output_dir / "misconceptions.jsonl"

        tasks = []
        for problem in problems:
            for _ in range(self.pairs_per_problem):
                tasks.append(self._generate_one(problem))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # PC-10: Close the shared vLLM session when done.
            if self._vllm_session is not None:
                await self._vllm_session.close()
                self._vllm_session = None

        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, MisconceptionPair):
                    # Convert to training conversation format
                    conversation = self._format_as_conversation(result)
                    await f.write(json.dumps(conversation) + "\n")
                    self._stats["generated"] += 1

        logger.info(
            f"Misconception generation complete: "
            f"{self._stats['generated']:,} pairs generated"
        )

    def _load_problems(self) -> list[dict]:
        """Load problems from raw data."""
        problems = []
        for jsonl_file in self.raw_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        problems.append(json.loads(line))
        return problems

    async def _generate_one(self, problem: dict) -> Optional[MisconceptionPair]:
        """Generate one misconception pair for a problem."""
        async with self._semaphore:
            statement = problem.get("statement", "")
            if not statement or len(statement) < 20:
                return None

            solutions = problem.get("solutions", [])
            if not solutions:
                return None

            # Use the first (highest-voted) solution as the correct one
            correct_solution = solutions[0].get("content", "")
            correct_answer = problem.get("answer", "")

            prompt = MISCONCEPTION_USER.format(
                problem_statement=statement,
                correct_solution=correct_solution[:1000],
                correct_answer=correct_answer or "varies",
            )

            response = await self._call_llm(MISCONCEPTION_SYSTEM, prompt, max_tokens=600)
            if not response:
                self._stats["failed"] += 1
                return None

            data = self._parse_json(response)
            if not data:
                self._stats["failed"] += 1
                return None

            quality = self._score_quality(data)

            return MisconceptionPair(
                problem_id=problem.get("problem_id", ""),
                problem_statement=statement,
                correct_answer=correct_answer,
                student_wrong_approach=data.get("student_wrong_approach", ""),
                student_wrong_answer=data.get("student_wrong_answer", ""),
                misconception_type=data.get("misconception_type", "unknown"),
                misconception_description=data.get("misconception_description", ""),
                corrective_question=data.get("corrective_question", ""),
                why_this_question=data.get("why_this_question", ""),
                quality_score=quality,
            )

    def _format_as_conversation(self, pair: MisconceptionPair) -> dict:
        """Format misconception pair as a training conversation."""
        user_content = (
            f"Problem: {pair.problem_statement}\n\n"
            f"My approach: {pair.student_wrong_approach}\n"
            f"My answer: {pair.student_wrong_answer}"
        )

        assistant_content = (
            f"I can see where your thinking led you. "
            f"{pair.corrective_question}"
        )

        return {
            "conversations": [
                {
                    "role": "system",
                    "content": (
                        "You are ProofCoach. A student has shown you their work. "
                        "Identify their misconception and ask a targeted corrective question."
                    ),
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "problem_id": pair.problem_id,
                "misconception_type": pair.misconception_type,
                "misconception_description": pair.misconception_description,
                "correct_answer": pair.correct_answer,
                "quality_score": pair.quality_score,
                "type": "misconception",
            },
        }

    def _score_quality(self, data: dict) -> float:
        """Score misconception pair quality."""
        score = 0.5

        if data.get("student_wrong_approach") and len(data["student_wrong_approach"]) > 50:
            score += 0.15
        if data.get("misconception_type") in MISCONCEPTION_CATEGORIES:
            score += 0.1
        if data.get("misconception_description") and len(data["misconception_description"]) > 30:
            score += 0.1
        if data.get("corrective_question") and "?" in data["corrective_question"]:
            score += 0.15
        if data.get("why_this_question"):
            score += 0.1

        return min(1.0, score)

    async def _call_llm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if self.backend == "claude":
            return await self._call_claude(system, user, max_tokens)
        return await self._call_vllm(system, user, max_tokens)

    async def _call_claude(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            resp = await self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.debug(f"Claude error: {e}")
            return None

    async def _call_vllm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if not self.vllm_urls:
            return None
        url = self.vllm_urls[self._vllm_index % len(self.vllm_urls)]
        self._vllm_index += 1

        # PC-10: Reuse shared session to avoid per-call TCP overhead.
        if self._vllm_session is None:
            self._vllm_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
            )

        try:
            async with self._vllm_session.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "/model",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.9,
                },
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', '')}"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"vLLM error: {e}")
        return None

    def _parse_json(self, text: str) -> Optional[dict]:
        import re, json
        match = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)
        try:
            return json.loads(text.strip())
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return None


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    generator = MisconceptionGenerator(
        raw_dir="data/raw",
        output_dir="data/synthesized/misconceptions",
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
    )
    asyncio.run(generator.generate_all())
