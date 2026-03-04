"""
khan_academy.py — Khan Academy internal API harvester for math exercises.

Khan Academy exposes a public GraphQL API at https://www.khanacademy.org/api/internal/graphql
and a legacy REST API at https://www.khanacademy.org/api/v1/.

Fetches:
  - Problem types with hints and step-by-step solutions
  - Curriculum-ordered problem sets by topic (pre-algebra → calculus)
  - Exercise metadata: hints, worked solutions, prerequisite skills

Output format:
  {
    exercise_id, title, topic, subtopic, difficulty,
    question_html, question_text,
    hints: [{step_number, text}],
    worked_solution, answer_type, answer,
    curriculum_order, prerequisites
  }

Target: 50k+ curriculum-ordered problem-hint-solution triples.

Usage:
    python discovery/khan_academy.py --topics algebra calculus
    python discovery/khan_academy.py --all
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from html.parser import HTMLParser

import aiofiles
import aiohttp
from loguru import logger

KA_API_V1 = "https://www.khanacademy.org/api/v1"
KA_GRAPHQL = "https://www.khanacademy.org/api/internal/graphql"
KA_CONTENT = "https://www.khanacademy.org"

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "khan_academy"

# Math topic slugs and their curriculum ordering
MATH_TOPICS = {
    # Foundational
    "early-math": {"order": 1, "level": "elementary"},
    "arithmetic": {"order": 2, "level": "elementary"},
    "pre-algebra": {"order": 3, "level": "middle"},
    "basic-geo": {"order": 4, "level": "middle"},
    "cc-sixth-grade-math": {"order": 5, "level": "middle"},
    "cc-seventh-grade-math": {"order": 6, "level": "middle"},
    "cc-eighth-grade-math": {"order": 7, "level": "middle"},
    # High school
    "algebra": {"order": 8, "level": "high_school"},
    "algebra2": {"order": 9, "level": "high_school"},
    "geometry": {"order": 10, "level": "high_school"},
    "trigonometry": {"order": 11, "level": "high_school"},
    "precalculus": {"order": 12, "level": "high_school"},
    "statistics-probability": {"order": 13, "level": "high_school"},
    # College
    "ap-calculus-ab": {"order": 14, "level": "college"},
    "ap-calculus-bc": {"order": 15, "level": "college"},
    "ap-statistics": {"order": 16, "level": "college"},
    "multivariable-calculus": {"order": 17, "level": "college"},
    "differential-equations": {"order": 18, "level": "college"},
    "linear-algebra": {"order": 19, "level": "college"},
}


class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def handle_entityref(self, name: str):
        entities = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "nbsp": " "}
        self._parts.append(entities.get(name, ""))

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def html_to_text(html: str) -> str:
    """Convert HTML content to plain text."""
    p = HTMLTextExtractor()
    try:
        p.feed(html)
        return p.get_text()
    except Exception:
        # Fallback: strip all tags
        return re.sub(r"<[^>]+>", " ", html).strip()


@dataclass
class KAHint:
    """A single hint step in a Khan Academy exercise."""

    step_number: int
    text: str
    has_image: bool = False


@dataclass
class KAExercise:
    """A Khan Academy math exercise with hints and solution."""

    exercise_id: str
    title: str
    topic: str
    subtopic: str
    difficulty: str  # "easy" | "medium" | "hard"
    curriculum_order: int
    level: str  # "elementary" | "middle" | "high_school" | "college"
    question_text: str
    hints: list[KAHint]
    worked_solution: str
    answer_type: str  # "multiple_choice" | "numeric" | "expression" | "free_response"
    answer: Optional[str]
    prerequisites: list[str]
    tags: list[str]
    source_url: str


class KhanAcademyHarvester:
    """
    Harvests Khan Academy math exercises via the public API.

    Uses a two-phase approach:
    1. Fetch topic tree to discover all exercises
    2. Fetch each exercise's hints and worked solutions
    """

    DELAY = 0.3

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        workers: int = 15,
        topics: Optional[list[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.topics = topics or list(MATH_TOPICS.keys())
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"exercises": 0, "errors": 0}

    async def _fetch_json(
        self, session: aiohttp.ClientSession, url: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Fetch JSON from Khan Academy API with retry."""
        headers = {
            "User-Agent": "Mozilla/5.0 (ProofCoach Research)",
            "Accept": "application/json",
        }
        for attempt in range(3):
            try:
                await asyncio.sleep(self.DELAY)
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        await asyncio.sleep(10 * (attempt + 1))
                    elif resp.status == 404:
                        return None
                    else:
                        logger.debug(f"HTTP {resp.status} for {url}")
                        return None
            except Exception as e:
                logger.debug(f"Fetch error: {url}: {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _get_topic_exercises(
        self, session: aiohttp.ClientSession, topic_slug: str
    ) -> list[dict]:
        """
        Fetch the list of exercises for a topic using the KA v1 API.

        GET /api/v1/topic/{slug}/exercises
        Returns list of exercise metadata.
        """
        url = f"{KA_API_V1}/topic/{topic_slug}/exercises"
        data = await self._fetch_json(session, url)
        if not data:
            return []

        exercises = []
        if isinstance(data, list):
            exercises = data
        elif isinstance(data, dict) and "exercises" in data:
            exercises = data["exercises"]

        return exercises

    async def _get_exercise_detail(
        self, session: aiohttp.ClientSession, exercise_slug: str
    ) -> Optional[dict]:
        """
        Fetch full exercise detail including items (question variants).

        GET /api/v1/exercises/{slug}
        GET /api/v1/exercises/{slug}/exercises_by_difficulty
        """
        url = f"{KA_API_V1}/exercises/{exercise_slug}"
        return await self._fetch_json(session, url)

    async def _get_exercise_items(
        self, session: aiohttp.ClientSession, exercise_id: str
    ) -> list[dict]:
        """
        Fetch actual question items for an exercise.

        GET /api/v1/exercises/{id}/assessment_items
        """
        url = f"{KA_API_V1}/exercises/{exercise_id}/assessment_items"
        data = await self._fetch_json(session, url, params={"lang": "en", "rand": "1"})
        if not data:
            return []
        if isinstance(data, list):
            return data[:10]  # Cap at 10 items per exercise
        return []

    def _parse_assessment_item(
        self,
        item: dict,
        exercise_meta: dict,
        topic: str,
        curriculum_order: int,
        level: str,
    ) -> Optional[KAExercise]:
        """Parse a single assessment item into a KAExercise."""
        item_data = item.get("item_data", item)
        if not item_data:
            return None

        # Parse the Perseus JSON format that KA uses
        perseus_data = None
        if isinstance(item_data, str):
            try:
                perseus_data = json.loads(item_data)
            except json.JSONDecodeError:
                return None
        elif isinstance(item_data, dict):
            perseus_data = item_data

        if not perseus_data:
            return None

        question = perseus_data.get("question", {})
        hints_raw = perseus_data.get("hints", [])

        # Extract question text
        content = question.get("content", "")
        question_text = html_to_text(content) if content else ""

        # Perseus uses LaTeX-style content with $formula$ for math
        if not question_text and "widgets" in question:
            # Try to extract from widgets
            for widget_id, widget in question.get("widgets", {}).items():
                if "content" in widget:
                    question_text = html_to_text(widget["content"])
                    break

        if not question_text or len(question_text) < 10:
            return None

        # Extract hints
        hints = []
        for i, hint in enumerate(hints_raw):
            hint_content = hint.get("content", "")
            if hint_content:
                hints.append(
                    KAHint(
                        step_number=i + 1,
                        text=html_to_text(hint_content),
                        has_image="{{image" in hint_content.lower(),
                    )
                )

        # Extract answer type and answer
        answer_type = "free_response"
        answer = None
        if "widgets" in question:
            for widget_id, widget in question.get("widgets", {}).items():
                wtype = widget.get("type", "")
                if wtype == "numeric-input":
                    answer_type = "numeric"
                    answer = (
                        str(
                            widget.get("props", {})
                            .get("answers", [{}])[0]
                            .get("value", "")
                        )
                        if widget.get("props", {}).get("answers")
                        else None
                    )
                elif wtype == "radio":
                    answer_type = "multiple_choice"
                    choices = widget.get("props", {}).get("choices", [])
                    correct = [
                        c.get("content", "") for c in choices if c.get("correct")
                    ]
                    answer = html_to_text(correct[0]) if correct else None
                elif wtype == "expression":
                    answer_type = "expression"

        # Exercise ID from KA
        exercise_id = (
            exercise_meta.get("id")
            or exercise_meta.get("internal_id")
            or exercise_meta.get("name", "unknown")
        )
        exercise_slug = exercise_meta.get("name") or exercise_meta.get(
            "slug", exercise_id
        )

        return KAExercise(
            exercise_id=f"{topic}_{exercise_id}_{item.get('sha', '')[:8]}",
            title=exercise_meta.get(
                "display_name", exercise_meta.get("title", exercise_slug)
            ),
            topic=topic,
            subtopic=exercise_meta.get("node_slug", ""),
            difficulty=exercise_meta.get("difficulty", "medium"),
            curriculum_order=curriculum_order,
            level=level,
            question_text=question_text,
            hints=hints,
            worked_solution=hints[-1].text if hints else "",
            answer_type=answer_type,
            answer=answer,
            prerequisites=exercise_meta.get("all_related_content", [])[:5],
            tags=exercise_meta.get("tags", []),
            source_url=f"{KA_CONTENT}/e/{exercise_slug}",
        )

    async def _process_exercise(
        self,
        session: aiohttp.ClientSession,
        exercise_meta: dict,
        topic: str,
        curriculum_order: int,
        level: str,
        output_file: Path,
    ) -> int:
        """Fetch and parse all items for a single exercise."""
        async with self._semaphore:
            exercise_id = exercise_meta.get("id") or exercise_meta.get("name", "")
            if not exercise_id:
                return 0

            items = await self._get_exercise_items(session, exercise_id)
            if not items:
                # Try the exercise itself as a single item
                detail = await self._get_exercise_detail(
                    session, exercise_meta.get("name", exercise_id)
                )
                if detail:
                    items = [detail]

            saved = 0
            async with aiofiles.open(output_file, "a") as f:
                for item in items:
                    exercise = self._parse_assessment_item(
                        item, exercise_meta, topic, curriculum_order, level
                    )
                    if exercise and exercise.question_text:
                        await f.write(json.dumps(asdict(exercise)) + "\n")
                        saved += 1

            self._stats["exercises"] += saved
            return saved

    async def _harvest_topic(self, session: aiohttp.ClientSession, topic: str) -> int:
        """Harvest all exercises for a single topic."""
        topic_config = MATH_TOPICS.get(topic, {"order": 99, "level": "unknown"})
        curriculum_order = topic_config["order"]
        level = topic_config["level"]

        output_file = self.output_dir / f"{topic}.jsonl"
        logger.info(f"Harvesting topic: {topic} (order={curriculum_order})")

        exercises = await self._get_topic_exercises(session, topic)
        if not exercises:
            # Try fetching sub-topics
            topic_data = await self._fetch_json(session, f"{KA_API_V1}/topic/{topic}")
            if topic_data:
                exercises = topic_data.get("exercises", [])

        if not exercises:
            logger.debug(f"No exercises found for topic: {topic}")
            return 0

        logger.info(f"  {topic}: {len(exercises)} exercises found")

        tasks = [
            self._process_exercise(
                session, ex, topic, curriculum_order, level, output_file
            )
            for ex in exercises
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = sum(r for r in results if isinstance(r, int))

        logger.info(f"  {topic}: {total} exercises saved")
        return total

    async def harvest_all(self) -> int:
        """Harvest all configured topics."""
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=30, limit_per_host=5),
        ) as session:
            total = 0
            # Process topics sequentially to respect rate limits
            for topic in self.topics:
                if topic not in MATH_TOPICS:
                    logger.warning(f"Unknown topic: {topic}")
                    continue
                n = await self._harvest_topic(session, topic)
                total += n

        logger.success(
            f"Khan Academy harvest complete: {self._stats['exercises']} exercises, "
            f"{self._stats['errors']} errors"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Harvest Khan Academy math exercises")
    parser.add_argument("--all", action="store_true", help="Harvest all math topics")
    parser.add_argument(
        "--topics", nargs="+", default=None, help=f"Topics: {list(MATH_TOPICS.keys())}"
    )
    parser.add_argument("--output-dir", default="data/raw/khan_academy")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--list", action="store_true", help="List available topics")
    args = parser.parse_args()

    if args.list:
        for slug, cfg in sorted(MATH_TOPICS.items(), key=lambda x: x[1]["order"]):
            print(f"  {cfg['order']:2d}. {slug:<30} ({cfg['level']})")
        raise SystemExit(0)

    topics = list(MATH_TOPICS.keys()) if args.all else args.topics

    harvester = KhanAcademyHarvester(
        output_dir=args.output_dir,
        workers=args.workers,
        topics=topics,
    )
    n = asyncio.run(harvester.harvest_all())
    print(f"\nTotal exercises harvested: {n:,}")
