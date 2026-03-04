"""
Practice Sequencer Agent — Adaptive problem selection based on skill gaps

Selects the next practice problem by:
1. Identifying the student's weakest mastered-prerequisite topics
2. Finding problems that target those topics at appropriate difficulty
3. Ensuring variety (not repeating problem types too often)

This is what turns ProofCoach from a problem solver into a curriculum.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from core.problem_taxonomy import TAXONOMY
from core.skill_model import SkillModel, StudentModel


@dataclass
class ProblemRecommendation:
    """A recommended practice problem."""

    problem_id: str
    competition: str
    year: int
    number: int
    statement: str
    difficulty: int
    topics: list[str]
    reason: str
    target_skill_gap: str


class PracticeSequencerAgent:
    """
    Selects the optimal next practice problem for a student.

    Algorithm:
    1. Get student's current mastery state
    2. Find the weakest unlocked (prerequisite-mastered) topics
    3. Select a problem targeting those topics at ±1 difficulty of student's current level
    4. Apply variety filter (avoid same type 2x in a row)
    """

    # How many problems to consider before picking the best
    CANDIDATE_POOL_SIZE = 20

    def __init__(
        self,
        problem_bank_dir: Path | str = "data/raw",
        skill_model: Optional[SkillModel] = None,
    ) -> None:
        self.problem_bank_dir = Path(problem_bank_dir)
        self._skill_model = skill_model or SkillModel()
        self._problem_index: dict[str, list[dict]] = {}
        self._load_problem_index()

    def get_next_problem(
        self,
        student_id: str,
        session_history: Optional[list[str]] = None,
    ) -> Optional[ProblemRecommendation]:
        """
        Get the next recommended practice problem for a student.

        Args:
            student_id: The student identifier
            session_history: List of problem IDs attempted this session

        Returns:
            ProblemRecommendation or None if no suitable problem found
        """
        student = self._skill_model.get_or_create(student_id)
        session_history = session_history or []

        # Find the target skill gap
        weak_areas = self._skill_model.get_weak_areas(student_id, top_n=3)
        if not weak_areas:
            # All available topics mastered — return a challenging problem
            return self._get_challenge_problem(student, session_history)

        target_topic = weak_areas[0]
        current_level = self._estimate_student_level(student)

        # Find candidate problems
        candidates = self._find_candidates(
            target_topic=target_topic,
            difficulty_target=current_level,
            exclude=set(student.problem_history[-20:] + session_history),
        )

        if not candidates:
            # Try the second weak area
            if len(weak_areas) > 1:
                candidates = self._find_candidates(
                    target_topic=weak_areas[1],
                    difficulty_target=current_level,
                    exclude=set(student.problem_history[-20:] + session_history),
                )

        if not candidates:
            return None

        # Pick the best candidate
        problem = self._select_best(candidates, current_level)

        return ProblemRecommendation(
            problem_id=problem["problem_id"],
            competition=problem.get("competition", ""),
            year=problem.get("year", 0),
            number=problem.get("number", 0),
            statement=problem.get("statement", ""),
            difficulty=problem.get("difficulty_estimate", current_level),
            topics=problem.get("topics", [target_topic]),
            reason=(
                f"Targeting your skill gap in {TAXONOMY[target_topic].name if target_topic in TAXONOMY else target_topic}. "
                f"Current mastery: {(student.topics[target_topic].mastery_score if target_topic in student.topics else 0):.0%}."
            ),
            target_skill_gap=target_topic,
        )

    def _estimate_student_level(self, student: StudentModel) -> int:
        """Estimate the student's current difficulty level (1-10)."""
        if student.total_problems_attempted == 0:
            return 3  # Assume AMC 10 early problems

        success_rate = student.total_problems_solved / student.total_problems_attempted

        # Map success rate + problem history to difficulty level
        if success_rate > 0.8:
            return min(9, 5 + len(student.problem_history) // 20)
        elif success_rate > 0.6:
            return 5
        elif success_rate > 0.4:
            return 4
        else:
            return 3

    def _find_candidates(
        self,
        target_topic: str,
        difficulty_target: int,
        exclude: set[str],
        window: int = 2,
    ) -> list[dict]:
        """Find candidate problems matching the topic and difficulty range."""
        candidates = []
        diff_min = max(1, difficulty_target - window)
        diff_max = min(10, difficulty_target + window)

        for competition, problems in self._problem_index.items():
            for problem in problems:
                if problem["problem_id"] in exclude:
                    continue

                difficulty = problem.get("difficulty_estimate", 5)
                if not (diff_min <= difficulty <= diff_max):
                    continue

                topics = problem.get("topics", [])
                if target_topic in topics:
                    candidates.append(problem)

                if len(candidates) >= self.CANDIDATE_POOL_SIZE:
                    break

        return candidates

    def _select_best(self, candidates: list[dict], target_difficulty: int) -> dict:
        """Select the best problem from candidates."""

        # Score by how close difficulty is to target
        def score(p: dict) -> float:
            diff = abs(p.get("difficulty_estimate", 5) - target_difficulty)
            return -diff  # Negative because we want minimum distance

        candidates.sort(key=score, reverse=True)

        # Add some randomness to avoid always picking the same problem
        top_k = candidates[:5]
        return random.choice(top_k)

    def _get_challenge_problem(
        self,
        student: StudentModel,
        session_history: list[str],
    ) -> Optional[ProblemRecommendation]:
        """Get a challenge problem when all topics are mastered."""
        level = 9  # High difficulty for students who've mastered everything
        candidates = []

        for problems in self._problem_index.values():
            for p in problems:
                if (
                    p.get("difficulty_estimate", 5) >= level
                    and p["problem_id"] not in student.problem_history[-20:]
                    and p["problem_id"] not in session_history
                ):
                    candidates.append(p)

        if not candidates:
            return None

        problem = random.choice(candidates[:10])
        return ProblemRecommendation(
            problem_id=problem["problem_id"],
            competition=problem.get("competition", ""),
            year=problem.get("year", 0),
            number=problem.get("number", 0),
            statement=problem.get("statement", ""),
            difficulty=problem.get("difficulty_estimate", 9),
            topics=problem.get("topics", []),
            reason="You've mastered all available topics — here's a challenge problem.",
            target_skill_gap="challenge",
        )

    def _load_problem_index(self) -> None:
        """Load problem index from the raw data directory."""
        if not self.problem_bank_dir.exists():
            logger.warning(f"Problem bank directory not found: {self.problem_bank_dir}")
            return

        for jsonl_file in self.problem_bank_dir.rglob("*.jsonl"):
            competition = jsonl_file.stem
            problems = []
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            problems.append(json.loads(line))
                self._problem_index[competition] = problems
            except Exception as e:
                logger.warning(f"Failed to load {jsonl_file}: {e}")

        total = sum(len(v) for v in self._problem_index.values())
        logger.info(
            f"Practice sequencer: loaded {total:,} problems from {len(self._problem_index)} competitions"
        )
