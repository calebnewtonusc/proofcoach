"""
socratic_synthesizer.py — Generate Socratic tutoring dialogues from problem + wrong attempt pairs.

Given (problem, student_wrong_answer), synthesizes a full Socratic dialogue where the tutor:
  1. Identifies the student's specific misconception
  2. Asks probing questions to guide insight (never gives away the answer directly)
  3. Guides the student to arrive at the correct solution themselves

Output format (per dialogue):
  {
    conversations: [{role, content}...],
    metadata: {problem_id, misconception, key_insight, dialogue_turns, ...}
  }

Supports both vLLM (bulk, cheap) and Claude (high quality) backends.

Usage:
    python synthesis/socratic_synthesizer.py --backend vllm \
        --vllm-urls http://localhost:8001 http://localhost:8002 \
        --input-dir data/raw/aops --output-dir data/synthesized/socratic
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for socratic_synthesizer. "
        "Install with: pip install aiofiles"
    ) from exc
import aiohttp
import anthropic
from loguru import logger


SOCRATIC_SYSTEM = """You are ProofCoach, an expert Socratic math tutor who specializes in competition mathematics (AMC, AIME, USAMO, IMO, Putnam).

Your teaching methodology:
1. NEVER give away the answer directly, even when explicitly asked
2. Identify the SPECIFIC misconception in the student's work — not generic mistakes
3. Ask probing questions that lead the student to discover the error themselves
4. When a student is stuck, provide a smaller hint that points toward the key insight
5. Celebrate small victories — when the student makes progress, acknowledge it
6. End dialogues when the student has truly understood, not just memorized the answer

A good Socratic dialogue:
- Has 4-7 turns (student attempt → tutor question → student thinking → tutor guidance → ...)
- Each tutor turn contains exactly one probing question (not multiple)
- Never contains the phrase "the answer is" or "therefore the solution is"
- Terminates when the student can explain WHY the solution works, not just WHAT it is

You output structured JSON for each dialogue."""

MISCONCEPTION_EXTRACTION_SYSTEM = """You are analyzing a student's mathematical work to identify their specific misconception.

You will be given:
1. A competition math problem
2. A student's incorrect or incomplete attempt

Your task: Identify the SPECIFIC mathematical misconception, not just "arithmetic error" or "wrong formula."

Good misconception descriptions:
- "Student confused AM-GM inequality direction: applied it to get a≥b but the constraint requires a≤b"
- "Student forgot that absolute value |x| ≥ 0, so |x| = -3 has no solution — treated it as a regular equation"
- "Student applied the quadratic formula but forgot to consider both roots; only took the positive root"

Output JSON: {"misconception": "...", "correct_approach_hint": "...", "key_insight": "..."}"""

DIALOGUE_GENERATION_PROMPT = """Generate a Socratic tutoring dialogue for this math problem.

Problem: {problem_statement}

Student's incorrect attempt:
{student_attempt}

Identified misconception: {misconception}
Key insight the student needs to discover: {key_insight}

Generate a realistic 4-7 turn dialogue where the tutor guides the student to the correct answer
through Socratic questioning. The student should do most of the thinking work.

Format as JSON:
{{
  "system": "You are ProofCoach, a Socratic math tutor.",
  "student_start": "{student_attempt}",
  "dialogue": [
    {{"role": "tutor", "content": "..."}},
    {{"role": "student", "content": "..."}},
    {{"role": "tutor", "content": "..."}},
    ...
  ],
  "key_insight": "...",
  "resolution": "student reached correct answer"
}}

Rules:
- Tutor turns: One question each, never give the answer
- Student turns: Show genuine thinking, including corrections as they happen
- Final student turn: Student explains why the answer is correct
- Total turns: 4-7"""


@dataclass
class WrongAttempt:
    """A student's incorrect attempt on a problem."""
    problem_id: str
    problem_statement: str
    correct_answer: Optional[str]
    student_attempt: str
    attempt_type: str    # "computation_error" | "wrong_formula" | "incomplete" | "misconception"


@dataclass
class SocraticDialogue:
    """A synthesized Socratic tutoring dialogue."""
    problem_id: str
    misconception: str
    key_insight: str
    turns: int
    conversation: dict     # Full conversations format
    quality_score: float


class SocraticSynthesizer:
    """
    Generates Socratic tutoring dialogues from problem-attempt pairs.

    Phase 1: Extract misconception from student attempt
    Phase 2: Generate full Socratic dialogue
    Phase 3: Quality score and filter
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 20,
        min_quality: float = 0.65,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self.min_quality = min_quality
        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_idx = 0
        self._stats = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "quality_filtered": 0,
        }

        if backend == "claude":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )

    def _next_vllm_url(self) -> str:
        # PC-11: Guard against ZeroDivisionError when vllm_urls is empty.
        if not self.vllm_urls:
            raise ValueError(
                "vllm_urls is empty — cannot select a vLLM endpoint. "
                "Pass --vllm-urls when using --backend vllm."
            )
        url = self.vllm_urls[self._vllm_idx % len(self.vllm_urls)]
        self._vllm_idx += 1
        return url

    async def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.8,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """Call vLLM or Claude based on backend setting."""
        if self.backend == "claude":
            return await self._call_claude(system, user, max_tokens)
        else:
            return await self._call_vllm(system, user, max_tokens, temperature, session)

    async def _call_claude(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            resp = await self._anthropic.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.debug(f"Claude error: {e}")
            return None

    async def _call_vllm(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        session: Optional[aiohttp.ClientSession],
    ) -> Optional[str]:
        if not self.vllm_urls or not session:
            return None
        url = self._next_vllm_url()
        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'synthesis')}"},
                timeout=aiohttp.ClientTimeout(total=90),
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

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        if not text:
            return None
        # Try code block
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            text = m.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Try finding JSON object
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _score_dialogue(self, dialogue_data: dict, n_turns: int) -> float:
        """Score dialogue quality 0.0-1.0."""
        score = 0.5

        turns = dialogue_data.get("dialogue", [])
        n = len(turns)

        # Good number of turns
        if 4 <= n <= 7:
            score += 0.15
        elif n < 3:
            return 0.1

        # Tutor turns contain questions
        tutor_turns = [t for t in turns if t.get("role") == "tutor"]
        if tutor_turns:
            q_ratio = sum(1 for t in tutor_turns if "?" in t.get("content", "")) / len(tutor_turns)
            score += 0.15 * q_ratio

        # No answer giveaway
        full_text = " ".join(t.get("content", "") for t in turns[:3]).lower()
        if any(phrase in full_text for phrase in ["the answer is", "therefore the solution is", "the answer equals"]):
            score -= 0.3

        # Has key insight
        if dialogue_data.get("key_insight") and len(dialogue_data["key_insight"]) > 20:
            score += 0.1

        # Student shows progression
        student_turns = [t for t in turns if t.get("role") == "student"]
        if len(student_turns) >= 2:
            # Check if later student turns are more complete/correct
            last_student = student_turns[-1].get("content", "")
            first_student = student_turns[0].get("content", "")
            if len(last_student) > len(first_student):
                score += 0.1

        return max(0.0, min(1.0, score))

    def _generate_wrong_attempt(self, problem: dict) -> Optional[WrongAttempt]:
        """
        Generate a plausible student wrong attempt from a problem record.

        If the problem already has wrong attempts (from forum discussions), use them.
        Otherwise synthesize a common misconception pattern.
        """
        problem_id = problem.get("problem_id", "unknown")
        statement = problem.get("statement", "")
        solutions = problem.get("solutions", [])
        answer = problem.get("answer")

        if not statement:
            return None

        # If there are multiple solutions, use later (harder) ones as wrong attempts
        wrong_attempt_text = ""
        attempt_type = "incomplete"

        # Check if we have forum solutions that might show wrong approaches
        if solutions and len(solutions) > 1:
            # The first solution is likely most standard; use a variation
            first_sol = solutions[0].get("content", "") if isinstance(solutions[0], dict) else str(solutions[0])
            # Truncate to simulate a partial/wrong attempt
            if len(first_sol) > 100:
                words = first_sol.split()
                cutoff = max(20, len(words) // 3)
                wrong_attempt_text = " ".join(words[:cutoff]) + "... [student gets stuck here]"
                attempt_type = "incomplete"

        if not wrong_attempt_text:
            # Generate a generic plausible wrong approach based on competition/topic
            competition = problem.get("competition", "")
            topics = problem.get("topics", [])

            if "number_theory" in topics:
                wrong_attempt_text = "I started by trying to factor the expression directly, but I think I'm missing something. I got that n divides the expression when n=1 and n=2, but my formula broke down for larger values..."
                attempt_type = "wrong_formula"
            elif "geometry" in topics:
                wrong_attempt_text = "I set up coordinates with the origin at the center, but when I computed the intersection point I got a negative distance, which doesn't make sense geometrically. I think I set up the equations wrong..."
                attempt_type = "computation_error"
            elif "combinatorics" in topics:
                wrong_attempt_text = "I used the multiplication principle and got n! / (n-k)!, but the answer key says the answer is much smaller. Did I overcount?"
                attempt_type = "wrong_formula"
            elif "algebra" in topics:
                wrong_attempt_text = "I moved everything to one side and got a quadratic equation. I got x = 3 as one solution, but when I plug it back in it doesn't work. I think I made an error squaring both sides..."
                attempt_type = "misconception"
            else:
                wrong_attempt_text = "I started working on this but I'm stuck. I tried setting up a system of equations but I have more unknowns than equations. Can you help me see what I'm missing?"
                attempt_type = "incomplete"

        return WrongAttempt(
            problem_id=problem_id,
            problem_statement=statement,
            correct_answer=str(answer) if answer else None,
            student_attempt=wrong_attempt_text,
            attempt_type=attempt_type,
        )

    async def _synthesize_one(
        self,
        attempt: WrongAttempt,
        output_file: Path,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[SocraticDialogue]:
        """Synthesize a Socratic dialogue for a single problem-attempt pair."""
        async with self._semaphore:
            self._stats["attempted"] += 1

            # Phase 1: Extract misconception
            misconception_prompt = (
                f"Problem: {attempt.problem_statement[:1500]}\n\n"
                f"Student attempt: {attempt.student_attempt}"
            )
            raw_misconception = await self._call_llm(
                MISCONCEPTION_EXTRACTION_SYSTEM,
                misconception_prompt,
                max_tokens=400,
                temperature=0.3,
                session=session,
            )
            if not raw_misconception:
                self._stats["failed"] += 1
                return None

            misconception_data = self._parse_json(raw_misconception)
            if not misconception_data:
                self._stats["failed"] += 1
                return None

            misconception = misconception_data.get("misconception", "unclear error")
            key_insight = misconception_data.get("key_insight", "")
            correct_hint = misconception_data.get("correct_approach_hint", "")

            # Phase 2: Generate Socratic dialogue
            dialogue_prompt = DIALOGUE_GENERATION_PROMPT.format(
                problem_statement=attempt.problem_statement[:1500],
                student_attempt=attempt.student_attempt,
                misconception=misconception,
                key_insight=key_insight,
            )

            raw_dialogue = await self._call_llm(
                SOCRATIC_SYSTEM,
                dialogue_prompt,
                max_tokens=2500,
                temperature=0.8,
                session=session,
            )
            if not raw_dialogue:
                self._stats["failed"] += 1
                return None

            dialogue_data = self._parse_json(raw_dialogue)
            if not dialogue_data:
                self._stats["failed"] += 1
                return None

            turns = dialogue_data.get("dialogue", [])
            quality = self._score_dialogue(dialogue_data, len(turns))

            if quality < self.min_quality:
                self._stats["quality_filtered"] += 1
                return None

            # Format as training conversation (multi-turn)
            conversation_turns = [
                {"role": "system", "content": dialogue_data.get("system", SOCRATIC_SYSTEM[:200])},
                {"role": "user", "content": f"Problem: {attempt.problem_statement}\n\nMy work: {attempt.student_attempt}"},
            ]

            # Add dialogue turns
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "tutor":
                    conversation_turns.append({"role": "assistant", "content": content})
                elif role == "student":
                    conversation_turns.append({"role": "user", "content": content})

            training_example = {
                "conversations": conversation_turns,
                "metadata": {
                    "problem_id": attempt.problem_id,
                    "misconception": misconception,
                    "key_insight": key_insight,
                    "attempt_type": attempt.attempt_type,
                    "dialogue_turns": len(turns),
                    "quality_score": quality,
                    "correct_answer": attempt.correct_answer,
                    "full_dialogue": turns,
                },
            }

            async with aiofiles.open(output_file, "a") as f:
                await f.write(json.dumps(training_example) + "\n")

            self._stats["succeeded"] += 1

            return SocraticDialogue(
                problem_id=attempt.problem_id,
                misconception=misconception,
                key_insight=key_insight,
                turns=len(turns),
                conversation=training_example,
                quality_score=quality,
            )

    def _load_problems(self) -> list[dict]:
        """Load all problems from input directory."""
        problems = []
        for jsonl_file in self.input_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            problems.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        logger.info(f"Loaded {len(problems)} problems from {self.input_dir}")
        return problems

    async def synthesize_all(self) -> int:
        """Synthesize Socratic dialogues for all problems."""
        problems = self._load_problems()

        # Generate wrong attempts
        attempts = []
        for prob in problems:
            attempt = self._generate_wrong_attempt(prob)
            if attempt:
                attempts.append(attempt)

        logger.info(f"Generated {len(attempts)} wrong attempts. Starting synthesis...")

        output_file = self.output_dir / "socratic_dialogues.jsonl"

        start = time.time()

        if self.backend == "vllm":
            async with aiohttp.ClientSession() as session:
                tasks = [self._synthesize_one(a, output_file, session) for a in attempts]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [self._synthesize_one(a, output_file) for a in attempts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start
        succeeded = sum(1 for r in results if isinstance(r, SocraticDialogue))

        logger.success(
            f"Socratic synthesis complete in {elapsed/60:.1f}m: "
            f"{self._stats['succeeded']} dialogues, "
            f"{self._stats['quality_filtered']} filtered, "
            f"{self._stats['failed']} failed"
        )
        return succeeded


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Synthesize Socratic tutoring dialogues")
    parser.add_argument("--input-dir", default="data/raw/aops")
    parser.add_argument("--output-dir", default="data/synthesized/socratic")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--min-quality", type=float, default=0.65)
    args = parser.parse_args()

    synthesizer = SocraticSynthesizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
        min_quality=args.min_quality,
    )
    n = asyncio.run(synthesizer.synthesize_all())
    print(f"\nTotal Socratic dialogues synthesized: {n:,}")
