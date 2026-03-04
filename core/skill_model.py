"""
Skill Model — Student knowledge state tracker

Maintains a probabilistic estimate of a student's mastery level
for each node in the problem taxonomy.

Used by:
  - practice_sequencer_agent.py — which problem to assign next
  - tutor_agent.py — how much scaffolding to provide
  - misconception_detector_agent.py — which misconceptions are likely

State is persisted to Redis (deployment) or a local JSON file (development).
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

from core.problem_taxonomy import TAXONOMY, get_unlockable_topics


@dataclass
class TopicMastery:
    """Mastery estimate for a single topic."""
    topic_id: str
    mastery_score: float = 0.0        # 0.0 = no evidence, 1.0 = fully mastered
    attempts: int = 0
    successes: int = 0
    last_attempt_ts: Optional[float] = None
    is_mastered: bool = False         # True when mastery_score > MASTERY_THRESHOLD
    is_unlocked: bool = False         # True when all prerequisites are mastered


@dataclass
class StudentModel:
    """
    Full student knowledge model.

    Tracks mastery per topic node, session history, and weak areas.
    """
    student_id: str
    created_ts: float = field(default_factory=time.time)
    last_active_ts: float = field(default_factory=time.time)
    topics: dict[str, TopicMastery] = field(default_factory=dict)
    session_count: int = 0
    total_problems_attempted: int = 0
    total_problems_solved: int = 0
    # Ordered list of problem IDs attempted (most recent last)
    problem_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Initialize all topic nodes
        for topic_id in TAXONOMY:
            if topic_id not in self.topics:
                node = TAXONOMY[topic_id]
                self.topics[topic_id] = TopicMastery(
                    topic_id=topic_id,
                    is_unlocked=len(node.prerequisites) == 0,
                )


MASTERY_THRESHOLD = 0.75    # Mastery score above which topic is "mastered"
DECAY_RATE = 0.005          # Daily decay rate for inactive topics


class SkillModel:
    """
    Manages student knowledge state with Bayesian-style updates.

    Uses an Elo-inspired rating system where:
    - Correct solution on hard problem → large mastery gain
    - Correct solution on easy problem → small mastery gain
    - Incorrect → mastery decreases proportionally to expected difficulty
    - Time decay → mastery slowly decreases if topic not practiced
    """

    def __init__(
        self,
        storage_dir: Optional[Path | str] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/student_models")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.redis_url = redis_url
        self._cache: dict[str, StudentModel] = {}

    def get_or_create(self, student_id: str) -> StudentModel:
        """Get existing student model or create a new one."""
        if student_id in self._cache:
            return self._cache[student_id]

        # Try to load from storage
        model = self._load(student_id)
        if model is None:
            model = StudentModel(student_id=student_id)
            logger.info(f"Created new student model: {student_id}")

        self._cache[student_id] = model
        return model

    def update(
        self,
        student_id: str,
        problem_id: str,
        topics: list[str],
        difficulty: int,
        solved: bool,
        time_taken_s: Optional[float] = None,
    ) -> StudentModel:
        """
        Update student model after a problem attempt.

        Args:
            student_id: Student identifier
            problem_id: Problem that was attempted
            topics: List of topic IDs this problem tests
            difficulty: Problem difficulty 1-10
            solved: Whether the student solved it correctly
            time_taken_s: Time taken in seconds (affects confidence update)
        """
        model = self.get_or_create(student_id)
        model.last_active_ts = time.time()
        model.total_problems_attempted += 1
        model.problem_history.append(problem_id)
        if len(model.problem_history) > 500:
            model.problem_history = model.problem_history[-500:]

        if solved:
            model.total_problems_solved += 1

        # Update mastery for each topic this problem tests
        for topic_id in topics:
            if topic_id not in model.topics:
                continue

            mastery = model.topics[topic_id]
            mastery.attempts += 1
            mastery.last_attempt_ts = time.time()

            if solved:
                mastery.successes += 1
                # Larger gain for harder problems
                gain = 0.1 + 0.05 * (difficulty - 5) / 5
                gain = max(0.05, min(0.25, gain))
                mastery.mastery_score = min(1.0, mastery.mastery_score + gain)
            else:
                # Larger loss for easier problems (should have known it)
                loss = 0.05 + 0.03 * (5 - difficulty) / 5
                loss = max(0.02, min(0.15, loss))
                mastery.mastery_score = max(0.0, mastery.mastery_score - loss)

            # Update mastery flag
            mastery.is_mastered = mastery.mastery_score >= MASTERY_THRESHOLD

        # Unlock new topics if prerequisites are now mastered
        mastered_topics = {
            tid for tid, tm in model.topics.items() if tm.is_mastered
        }
        newly_unlockable = get_unlockable_topics(mastered_topics)
        for topic_id in newly_unlockable:
            if topic_id in model.topics:
                model.topics[topic_id].is_unlocked = True

        self._save(model)
        return model

    def get_weak_areas(self, student_id: str, top_n: int = 5) -> list[str]:
        """
        Get the topics with the lowest mastery that are currently unlocked.

        Used by the practice sequencer to target skill gaps.
        """
        model = self.get_or_create(student_id)
        unlocked = [
            (tid, tm) for tid, tm in model.topics.items()
            if tm.is_unlocked and not tm.is_mastered
        ]
        unlocked.sort(key=lambda x: x[1].mastery_score)
        return [tid for tid, _ in unlocked[:top_n]]

    def get_mastered_topics(self, student_id: str) -> set[str]:
        """Get the set of mastered topic IDs for a student."""
        model = self.get_or_create(student_id)
        return {tid for tid, tm in model.topics.items() if tm.is_mastered}

    def get_unlocked_topics(self, student_id: str) -> set[str]:
        """Get the set of unlocked (accessible) topic IDs for a student."""
        model = self.get_or_create(student_id)
        return {tid for tid, tm in model.topics.items() if tm.is_unlocked}

    def get_mastery_summary(self, student_id: str) -> dict:
        """Get a summary of student mastery for display."""
        model = self.get_or_create(student_id)

        by_category: dict[str, dict] = {}
        for topic_id, mastery in model.topics.items():
            node = TAXONOMY.get(topic_id)
            if not node:
                continue
            cat = node.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "unlocked": 0, "mastered": 0}
            by_category[cat]["total"] += 1
            if mastery.is_unlocked:
                by_category[cat]["unlocked"] += 1
            if mastery.is_mastered:
                by_category[cat]["mastered"] += 1

        return {
            "student_id": student_id,
            "total_problems": model.total_problems_attempted,
            "solved_rate": (
                model.total_problems_solved / max(model.total_problems_attempted, 1)
            ),
            "session_count": model.session_count,
            "by_category": by_category,
            "weak_areas": self.get_weak_areas(student_id),
        }

    def apply_time_decay(self, student_id: str) -> None:
        """Apply time-based decay to mastery scores for inactive topics."""
        model = self.get_or_create(student_id)
        now = time.time()

        for mastery in model.topics.values():
            if mastery.last_attempt_ts is None:
                continue
            days_since = (now - mastery.last_attempt_ts) / 86400
            if days_since > 7:  # Only decay after a week of inactivity
                decay = DECAY_RATE * days_since
                mastery.mastery_score = max(0.0, mastery.mastery_score - decay)
                if mastery.mastery_score < MASTERY_THRESHOLD:
                    mastery.is_mastered = False

        self._save(model)

    def _save(self, model: StudentModel) -> None:
        """Save student model to disk."""
        path = self.storage_dir / f"{model.student_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(model), f, indent=2)

    def _load(self, student_id: str) -> Optional[StudentModel]:
        """Load student model from disk."""
        path = self.storage_dir / f"{student_id}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            model = StudentModel(student_id=data["student_id"])
            model.created_ts = data.get("created_ts", time.time())
            model.last_active_ts = data.get("last_active_ts", time.time())
            model.session_count = data.get("session_count", 0)
            model.total_problems_attempted = data.get("total_problems_attempted", 0)
            model.total_problems_solved = data.get("total_problems_solved", 0)
            model.problem_history = data.get("problem_history", [])

            # Restore topic mastery
            for tid, tm_data in data.get("topics", {}).items():
                if tid in TAXONOMY:
                    model.topics[tid] = TopicMastery(**tm_data)

            return model
        except Exception as e:
            logger.warning(f"Failed to load model for {student_id}: {e}")
            return None
