"""
Tutor Agent — Socratic tutoring via the trained ProofCoach model

The primary agent: takes a problem + student work and returns
a targeted Socratic question or hint at the appropriate level.

Hint levels:
  1 = Just a nudge ("What property does this number have?")
  2 = Direction hint ("Think about parity...")
  3 = Approach hint ("Consider using modular arithmetic here.")
  4 = Method hint ("Try working mod 4 and see what happens to each term.")
  5 = Near-solution ("Express each term as (2k+1)^2 and reduce mod 4.")
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.lean4_interface import Lean4Interface
from synthesis.prompts import PROOFCOACH_SYSTEM


@dataclass
class TutoringSession:
    """A single tutoring session state."""

    session_id: str
    problem: str
    student_id: Optional[str] = None
    turns: list[dict] = field(default_factory=list)
    hint_level: int = 1
    solved: bool = False
    start_ts: float = field(default_factory=time.time)


@dataclass
class TutoringResponse:
    """Response from the tutor agent."""

    question: str
    hint_level: int
    verified_steps: list[str]
    approach_hint: Optional[str]
    next_problem_suggestion: Optional[str]
    lean4_claims: list[dict]
    session_id: str


SYSTEM_PROMPT = PROOFCOACH_SYSTEM


class TutorAgent:
    """
    Socratic tutor agent using the trained ProofCoach model.

    Generates targeted questions based on:
    - Student's work attempt
    - Current hint level (escalates if student is stuck)
    - Topics involved in the problem
    - Student's mastery model (if available)
    """

    def __init__(
        self,
        model_path: str = "checkpoints/proofcoach-final",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        lean4_verify: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lean4_verify = lean4_verify

        self._sessions: dict[str, TutoringSession] = {}

        logger.info(f"Loading tutor model from {model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_path, trust_remote_code=True
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except (ValueError, ImportError):
            logger.warning(
                "flash_attention_2 not available; falling back to eager attention"
            )
            self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="eager",
                trust_remote_code=True,
            )
        self._model.eval()

        if lean4_verify:
            self._lean4 = Lean4Interface(
                timeout=int(os.getenv("LEAN4_TIMEOUT", 10)),
                simulated=os.getenv("LEAN4_SIMULATED", "0") == "1",
            )
        else:
            self._lean4 = None

        logger.info("TutorAgent ready")

    def tutor(
        self,
        problem: str,
        student_work: Optional[str],
        session_id: str,
        student_id: Optional[str] = None,
        force_hint_level: Optional[int] = None,
    ) -> dict:
        """
        Generate a tutoring response for a student's work.

        Args:
            problem: The math problem statement
            student_work: The student's current attempt (None if starting fresh)
            session_id: Unique session identifier
            student_id: Optional student identifier for skill model lookup
            force_hint_level: Override the adaptive hint level

        Returns:
            Dict with question, hint_level, verified_steps, etc.
        """
        session = self._get_or_create_session(session_id, problem, student_id)

        # Determine hint level
        if force_hint_level is not None:
            hint_level = force_hint_level
        else:
            hint_level = self._determine_hint_level(session, student_work)

        session.hint_level = hint_level

        # Build the conversation context
        messages = self._build_messages(session, student_work, hint_level)

        # Generate response
        raw_response = self._generate(messages)

        # Extract structured components
        question = self._extract_question(raw_response)
        verified_steps = self._extract_and_verify_claims(raw_response)
        next_problem = self._extract_next_problem_suggestion(raw_response)
        approach_hint = self._extract_approach_hint(raw_response, hint_level)

        # Update session
        session.turns.append(
            {
                "role": "student",
                "content": student_work or "",
                "hint_level": hint_level,
            }
        )
        session.turns.append(
            {
                "role": "tutor",
                "content": raw_response,
            }
        )

        return {
            "question": question,
            "hint_level": hint_level,
            "verified_steps": verified_steps,
            "approach_hint": approach_hint,
            "next_problem_suggestion": next_problem,
            "lean4_claims": [
                {"claim": step, "verified": True} for step in verified_steps
            ],
            "session_id": session_id,
            "full_response": raw_response,
        }

    def _get_or_create_session(
        self, session_id: str, problem: str, student_id: Optional[str]
    ) -> TutoringSession:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = TutoringSession(
                session_id=session_id,
                problem=problem,
                student_id=student_id,
            )
        return self._sessions[session_id]

    def _determine_hint_level(
        self, session: TutoringSession, student_work: Optional[str]
    ) -> int:
        """
        Adaptively determine hint level based on session history.

        Escalates if the student has been stuck for multiple turns.
        """
        if not student_work:
            return 1  # Fresh start — minimal hint

        n_turns = len(session.turns)

        # Detect if student is stuck (repeated similar attempts)
        if n_turns >= 4:
            # Check for progress indicators in recent turns
            recent_student = [
                t["content"]
                for t in session.turns[-4:]
                if t["role"] == "student" and t.get("content")
            ]
            if len(recent_student) >= 2 and self._looks_stuck(recent_student):
                return min(5, session.hint_level + 1)

        # Gradually escalate over the session
        if n_turns > 10:
            return 5
        elif n_turns > 6:
            return min(4, session.hint_level + 1)

        return session.hint_level

    def _looks_stuck(self, recent_attempts: list[str]) -> bool:
        """Check if the student appears stuck (similar answers across turns)."""
        if len(recent_attempts) < 2:
            return False
        # Simple heuristic: if attempts are very similar, student is stuck
        last = recent_attempts[-1].lower()
        second_last = recent_attempts[-2].lower()
        common = sum(1 for c in last if c in second_last)
        similarity = common / max(len(last), 1)
        return similarity > 0.8

    def _build_messages(
        self,
        session: TutoringSession,
        student_work: Optional[str],
        hint_level: int,
    ) -> list[dict]:
        """Build the message history for the model."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add session history (last 6 turns to stay within context)
        for turn in session.turns[-6:]:
            role = "user" if turn["role"] == "student" else "assistant"
            messages.append({"role": role, "content": turn["content"]})

        # Add current student message
        user_content = f"Problem: {session.problem}"
        if student_work:
            user_content += f"\n\nMy work: {student_work}"
        else:
            user_content += "\n\nI'm not sure where to start."

        user_content += f"\n\n[Hint level requested: {hint_level}/5]"

        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate(self, messages: list[dict]) -> str:
        """Generate a response using the tutor model."""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_question(self, response: str) -> str:
        """Extract the main Socratic question from the response."""
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())

        # Find the first sentence ending in "?"
        for sent in sentences:
            if sent.endswith("?"):
                return sent

        # Fallback: return the last sentence
        return sentences[-1] if sentences else response[:200]

    def _extract_and_verify_claims(self, response: str) -> list[str]:
        """Extract and Lean 4 verify any mathematical claims in the response."""
        if not self._lean4:
            return []

        claims = self._lean4.extract_claims_from_dialogue(response)
        if not claims:
            return []

        results = self._lean4.verify_batch(claims)
        return [r.theorem for r in results if r.success]

    def _extract_approach_hint(self, response: str, hint_level: int) -> Optional[str]:
        """Extract any approach hint from the response (only at level 3+)."""
        if hint_level < 3:
            return None

        # Look for technique keywords
        techniques = [
            "modular arithmetic",
            "induction",
            "AM-GM",
            "Cauchy-Schwarz",
            "pigeonhole",
            "generating functions",
            "vieta",
            "induction",
            "contradiction",
            "casework",
            "constructive",
        ]
        response_lower = response.lower()
        for tech in techniques:
            if tech in response_lower:
                return f"Consider: {tech}"
        return None

    def _extract_next_problem_suggestion(self, response: str) -> Optional[str]:
        """Extract any suggested next practice problem from the response."""
        # Look for patterns like "Try: AMC 2019 12A Problem 15"
        match = re.search(
            r"(?:try|next problem|practice)[:\s]+([A-Z\d\s/]+Problem[:\s]\d+)",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None
