"""
ProofCoach Synthesis Pipeline — Bulk async synthesis

Generates Socratic tutoring dialogues from (problem, solution) pairs
using either Claude (Anthropic API) or vLLM (local Qwen2.5-72B).

Usage:
    python synthesis/synthesize_bulk.py --backend claude --workers 20
    python synthesis/synthesize_bulk.py --backend vllm --vllm-urls http://localhost:8001 http://localhost:8002
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for synthesize_bulk. Install with: pip install aiofiles"
    ) from exc
import aiohttp
import anthropic
from loguru import logger

from synthesis.prompts import (
    TEACHING_SYSTEM,
    TEACHING_USER,
    EXTRACT_APPROACH_SYSTEM,
    EXTRACT_APPROACH_USER,
    LEAN4_EXTRACT_SYSTEM,
    LEAN4_EXTRACT_USER,
)


@dataclass
class SynthesisResult:
    """Result of synthesizing one tutoring dialogue."""

    problem_id: str
    approach_name: str
    conversation: dict
    quality_score: float
    lean4_claims: list[str]
    success: bool
    error: Optional[str] = None


class SynthesisPipeline:
    """
    Async pipeline for bulk synthesis of tutoring dialogues.

    For each (problem, solution_approach) pair:
    1. Extract the key insight and approach name
    2. Generate a Socratic tutoring dialogue
    3. Extract and validate Lean 4 claims
    4. Score quality
    5. Write to output JSONL
    """

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 20,
        min_quality_score: float = 0.6,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self.min_quality_score = min_quality_score

        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_index = 0  # round-robin index for vLLM instances

        if backend == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running the synthesis pipeline."
                )
            self._claude_client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self._claude_client = None

        self._stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "quality_filtered": 0,
        }
        # PC-9: Shared aiohttp session for all vLLM calls (avoids per-call overhead).
        self._vllm_session: Optional[aiohttp.ClientSession] = None

    async def synthesize_all(self) -> None:
        """Synthesize dialogues for all problems in raw_dir."""
        problems = self._load_all_problems()
        logger.info(
            f"Synthesizing dialogues for {len(problems):,} (problem, approach) pairs..."
        )

        start_time = time.time()

        try:
            tasks = [
                self._synthesize_one(prob, approach) for prob, approach in problems
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # PC-9: Close the shared vLLM session when done.
            if self._vllm_session is not None:
                await self._vllm_session.close()
                self._vllm_session = None

        elapsed = time.time() - start_time
        logger.info(
            f"Synthesis complete in {elapsed / 60:.1f}m: "
            f"{self._stats['success']:,} success, "
            f"{self._stats['failed']:,} failed, "
            f"{self._stats['quality_filtered']:,} filtered"
        )

    def _load_all_problems(self) -> list[tuple[dict, dict]]:
        """Load all (problem, solution_approach) pairs from raw data."""
        pairs = []
        for jsonl_file in self.raw_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    problem = json.loads(line)
                    solutions = problem.get("solutions", [])
                    # Create one synthesis task per solution approach
                    for solution in solutions:
                        pairs.append((problem, solution))
        return pairs

    async def _synthesize_one(
        self, problem: dict, solution: dict
    ) -> Optional[SynthesisResult]:
        """Synthesize a single tutoring dialogue from a problem + solution approach."""
        async with self._semaphore:
            try:
                # Step 1: Extract the approach and key insight
                approach_data = await self._extract_approach(problem, solution)
                if not approach_data:
                    self._stats["failed"] += 1
                    return None

                # Step 2: Generate the Socratic dialogue
                dialogue = await self._generate_dialogue(
                    problem, solution, approach_data
                )
                if not dialogue:
                    self._stats["failed"] += 1
                    return None

                # Step 3: Extract Lean 4 claims from the dialogue
                lean4_claims = await self._extract_lean4_claims(dialogue)

                # Step 4: Score quality
                quality_score = self._score_quality(dialogue, approach_data)

                if quality_score < self.min_quality_score:
                    self._stats["quality_filtered"] += 1
                    return None

                # Step 5: Format as ShareGPT conversation
                conversation = self._format_conversation(
                    problem, solution, approach_data, dialogue, lean4_claims
                )

                result = SynthesisResult(
                    problem_id=problem.get("problem_id", "unknown"),
                    approach_name=approach_data.get("approach_name", "unknown"),
                    conversation=conversation,
                    quality_score=quality_score,
                    lean4_claims=lean4_claims,
                    success=True,
                )

                # Write immediately (don't buffer in memory)
                output_file = (
                    self.output_dir
                    / f"{problem.get('competition', 'unknown').lower()}.jsonl"
                )
                async with aiofiles.open(output_file, "a") as f:
                    await f.write(json.dumps(conversation) + "\n")

                self._stats["success"] += 1
                self._stats["processed"] += 1

                if self._stats["processed"] % 100 == 0:
                    logger.info(f"Progress: {self._stats['processed']:,} processed")

                return result

            except Exception as e:
                logger.debug(
                    f"Error synthesizing {problem.get('problem_id', '?')}: {e}"
                )
                self._stats["failed"] += 1
                return None

    async def _extract_approach(self, problem: dict, solution: dict) -> Optional[dict]:
        """Extract approach name, key insight, student stuck point."""
        prompt = EXTRACT_APPROACH_USER.format(
            problem_statement=problem.get("statement", ""),
            solution_text=solution.get("content", ""),
        )
        response = await self._call_llm(EXTRACT_APPROACH_SYSTEM, prompt, max_tokens=400)
        if not response:
            return None
        return self._parse_json_response(response)

    async def _generate_dialogue(
        self, problem: dict, solution: dict, approach_data: dict
    ) -> Optional[str]:
        """Generate the full Socratic tutoring dialogue."""
        solutions = problem.get("solutions", [])
        approach_number = next(
            (
                i + 1
                for i, s in enumerate(solutions)
                if s.get("post_id") == solution.get("post_id")
            ),
            1,
        )

        prompt = TEACHING_USER.format(
            problem_statement=problem.get("statement", ""),
            approach_number=approach_number,
            total_approaches=len(solutions),
            approach_name=approach_data.get("approach_name", "standard"),
            solution_text=solution.get("content", ""),
            key_insight=approach_data.get("key_insight", ""),
            student_stuck_point=approach_data.get("student_stuck_point", ""),
        )

        return await self._call_llm(TEACHING_SYSTEM, prompt, max_tokens=1500)

    async def _extract_lean4_claims(self, dialogue_text: str) -> list[str]:
        """Extract verifiable Lean 4 claims from the dialogue."""
        response = await self._call_llm(
            LEAN4_EXTRACT_SYSTEM,
            LEAN4_EXTRACT_USER.format(dialogue_text=dialogue_text[:3000]),
            max_tokens=500,
        )
        if not response:
            return []

        data = self._parse_json_response(response)
        if not data:
            return []

        claims = data.get("claims", [])
        return [
            c.get("lean4_proposition", "") for c in claims if c.get("lean4_proposition")
        ]

    def _score_quality(self, dialogue_text: str, approach_data: dict) -> float:
        """Score dialogue quality heuristically (0.0 to 1.0)."""
        if not dialogue_text:
            return 0.0

        score = 0.5  # base score

        # Check it's valid JSON with expected structure
        data = self._parse_json_response(dialogue_text)
        if not data:
            return 0.2

        dialogue_turns = data.get("dialogue", [])

        # Good number of turns (4-7)
        n_turns = len(dialogue_turns)
        if 4 <= n_turns <= 7:
            score += 0.15
        elif n_turns < 2:
            return 0.1  # Too short

        # Check tutor turns contain questions (?)
        tutor_turns = [t for t in dialogue_turns if t.get("role") == "tutor"]
        question_ratio = sum(
            1 for t in tutor_turns if "?" in t.get("content", "")
        ) / max(len(tutor_turns), 1)
        score += 0.15 * question_ratio

        # Check it doesn't just give the answer directly
        full_text = " ".join(t.get("content", "") for t in dialogue_turns[:2])
        if any(
            phrase in full_text.lower()
            for phrase in [
                "the answer is",
                "therefore the answer",
                "so the final answer",
            ]
        ):
            score -= 0.3

        # Has a key insight
        if data.get("key_insight") and len(data["key_insight"]) > 20:
            score += 0.1

        # Has Lean 4 claims
        if data.get("lean4_claims"):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _format_conversation(
        self,
        problem: dict,
        solution: dict,
        approach_data: dict,
        dialogue_text: str,
        lean4_claims: list[str],
    ) -> dict:
        """Format as ShareGPT training conversation."""
        data = self._parse_json_response(dialogue_text) or {}

        # Build the conversation from the dialogue
        user_content = (
            f"Problem: {problem.get('statement', '')}\n\n"
            f"My work so far: {data.get('student_start', 'I am not sure where to begin.')}"
        )

        # Build the assistant response from first tutor turn
        dialogue = data.get("dialogue", [])
        first_tutor = next(
            (t["content"] for t in dialogue if t.get("role") == "tutor"), ""
        )

        if not first_tutor:
            first_tutor = "Let me ask you a question to help you think through this..."

        return {
            "conversations": [
                {
                    "role": "system",
                    "content": data.get(
                        "system", "You are ProofCoach, a Socratic math tutor."
                    ),
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": first_tutor},
            ],
            "metadata": {
                "problem_id": problem.get("problem_id", ""),
                "competition": problem.get("competition", ""),
                "year": problem.get("year"),
                "number": problem.get("number"),
                "approach_name": approach_data.get("approach_name", ""),
                "key_insight": approach_data.get("key_insight", ""),
                "difficulty": problem.get(
                    "difficulty", approach_data.get("difficulty", 5)
                ),
                "topics": problem.get("topics", []),
                "lean4_claims": lean4_claims,
                "full_dialogue": data.get("dialogue", []),
            },
        }

    async def _call_llm(
        self, system: str, user: str, max_tokens: int = 1000
    ) -> Optional[str]:
        """Call LLM (Claude or vLLM) and return text response."""
        if self.backend == "claude":
            return await self._call_claude(system, user, max_tokens)
        else:
            return await self._call_vllm(system, user, max_tokens)

    async def _call_claude(
        self, system: str, user: str, max_tokens: int
    ) -> Optional[str]:
        """Call Anthropic Claude API."""
        try:
            response = await self._claude_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except Exception as e:
            logger.debug(f"Claude API error: {e}")
            return None

    async def _call_vllm(
        self, system: str, user: str, max_tokens: int
    ) -> Optional[str]:
        """Call local vLLM instance (round-robin load balancing)."""
        if not self.vllm_urls:
            return None

        url = self.vllm_urls[self._vllm_index % len(self.vllm_urls)]
        self._vllm_index += 1

        api_key = os.getenv("VLLM_API_KEY", "proofcoach_synth_key")

        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }

        # PC-9: Reuse the shared session stored on the instance instead of
        # creating a new aiohttp.ClientSession per call. A new session on every
        # call disables connection pooling, exhausts file descriptors under
        # concurrency, and incurs TCP handshake overhead for every request.
        if self._vllm_session is None:
            self._vllm_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
            )

        try:
            async with self._vllm_session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.debug(f"vLLM error {resp.status}")
                    return None
        except Exception as e:
            logger.debug(f"vLLM call failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        import re

        # Try to find JSON in code block first
        match = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
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
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/synthesized/teaching")
    args = parser.parse_args()

    pipeline = SynthesisPipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
    )

    asyncio.run(pipeline.synthesize_all())
