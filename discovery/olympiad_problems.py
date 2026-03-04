"""
olympiad_problems.py — Exhaustive competition math problem harvester.

Downloads from:
  - IMO official problems archive (1959-2024): https://www.imo-official.org/problems.aspx
  - AIME problems from AoPS wiki (1983-2024)
  - USAMO problems (1972-2024)
  - Putnam A/B problems (1938-2024)
  - HMMT, ARML, Harvard-MIT competitions

Creates records:
  {competition, year, problem_number, statement, solution, difficulty, answer_type, source_url}

Target: 25k+ problem-solution pairs across all competitions.

Usage:
    python discovery/olympiad_problems.py --all
    python discovery/olympiad_problems.py --competition IMO AIME_I USAMO
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

AOPS_WIKI = "https://artofproblemsolving.com/wiki/index.php"
IMO_OFFICIAL = "https://www.imo-official.org"
OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "olympiads"


@dataclass
class OlympiadProblem:
    """A competition math problem with full solution."""
    problem_id: str
    competition: str
    year: int
    number: int
    statement: str
    solutions: list[str]          # May have multiple solutions
    answer: Optional[str]         # Numerical answer for AMC/AIME
    answer_type: str              # "multiple_choice" | "integer" | "proof"
    difficulty: int               # 1-10
    source_url: str
    topics: list[str]


# Competition configuration
COMPETITIONS = {
    "AMC_8": {
        "years": list(range(1999, 2026)),
        "problems": 25,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_AMC_8_Problems/Problem_{n}",
    },
    "AMC_10A": {
        "years": list(range(2002, 2026)),
        "problems": 30,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_AMC_10A_Problems/Problem_{n}",
    },
    "AMC_10B": {
        "years": list(range(2002, 2026)),
        "problems": 30,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_AMC_10B_Problems/Problem_{n}",
    },
    "AMC_12A": {
        "years": list(range(2002, 2026)),
        "problems": 30,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_AMC_12A_Problems/Problem_{n}",
    },
    "AMC_12B": {
        "years": list(range(2002, 2026)),
        "problems": 30,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_AMC_12B_Problems/Problem_{n}",
    },
    "AIME_I": {
        "years": list(range(1983, 2026)),
        "problems": 15,
        "answer_type": "integer",
        "wiki_template": "{year}_AIME_I_Problems/Problem_{n}",
    },
    "AIME_II": {
        "years": list(range(2000, 2026)),
        "problems": 15,
        "answer_type": "integer",
        "wiki_template": "{year}_AIME_II_Problems/Problem_{n}",
    },
    "USAMO": {
        "years": list(range(1972, 2026)),
        "problems": 6,
        "answer_type": "proof",
        "wiki_template": "{year}_USAMO_Problems/Problem_{n}",
    },
    "USAJMO": {
        "years": list(range(2010, 2026)),
        "problems": 6,
        "answer_type": "proof",
        "wiki_template": "{year}_USAJMO_Problems/Problem_{n}",
    },
    "IMO": {
        "years": list(range(1959, 2026)),
        "problems": 6,
        "answer_type": "proof",
        "wiki_template": "{year}_IMO_Problems/Problem_{n}",
    },
    "HMMT_Nov": {
        "years": list(range(2005, 2025)),
        "problems": 30,
        "answer_type": "multiple_choice",
        "wiki_template": "{year}_HMMT_November_Problems",
    },
    "Putnam": {
        "years": list(range(1938, 2025)),
        "problems": 12,     # 6 A + 6 B
        "answer_type": "proof",
        "wiki_template": "{year}_Putnam_Problems",
    },
}


def _difficulty(competition: str, number: int) -> int:
    """Estimate difficulty 1-10 based on competition and problem position."""
    if "AMC_8" in competition:
        return max(1, min(5, 1 + number // 5))
    elif "AMC_10" in competition:
        return max(2, min(7, 1 + number // 5))
    elif "AMC_12" in competition:
        return max(3, min(8, 1 + number // 5))
    elif "AIME" in competition:
        return max(4, min(9, 3 + number // 2))
    elif competition in ("USAMO", "USAJMO"):
        return [7, 8, 10, 7, 8, 10][min(number - 1, 5)]
    elif competition == "IMO":
        return [7, 8, 10, 7, 9, 10][min(number - 1, 5)]
    elif "HMMT" in competition:
        return max(4, min(9, 3 + number // 5))
    elif competition == "Putnam":
        return max(7, min(10, 6 + number // 3))
    return 5


def _infer_topics(statement: str, solutions: list[str]) -> list[str]:
    """Infer math topics from problem content."""
    text = (statement + " ".join(solutions)).lower()
    topic_keywords = {
        "number_theory": ["prime", "divisib", "modular", "gcd", "lcm", "congruent", "remainder"],
        "combinatorics": ["count", "combinat", "permut", "choose", "arrange", "pigeonhole"],
        "algebra": ["polynomial", "equation", "inequalit", "sequence", "function", "factor"],
        "geometry": ["triangle", "circle", "angle", "perpendicular", "parallel", "chord", "tangent"],
        "probability": ["probabilit", "expected", "random", "event"],
        "calculus": ["limit", "derivative", "integral", "continuous"],
    }
    return [t for t, kws in topic_keywords.items() if any(kw in text for kw in kws)] or ["general"]


class OlympiadHarvester:
    """
    Async harvester for competition math problems from AoPS wiki and official sources.
    """

    DELAY = 0.25  # seconds between requests

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        workers: int = 10,
        competitions: Optional[list[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.competitions = competitions or list(COMPETITIONS.keys())
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"problems": 0, "errors": 0}

    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch URL with retry and polite delay."""
        for attempt in range(3):
            try:
                await asyncio.sleep(self.DELAY)
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 404:
                        return None
                    elif resp.status == 429:
                        await asyncio.sleep(10 * (attempt + 1))
                    else:
                        return None
            except Exception as e:
                if attempt == 2:
                    logger.debug(f"Fetch failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    def _extract_statement(self, soup: BeautifulSoup) -> str:
        """Extract problem statement from AoPS wiki page."""
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return ""

        parts = []
        for elem in content.children:
            if not hasattr(elem, "name"):
                continue
            if elem.name in ("h2", "h3"):
                heading = elem.get_text()
                if any(kw in heading for kw in ("Solution", "See Also", "Problem", "Shortlist")):
                    break
            if elem.name in ("p", "dl", "ul", "ol", "div"):
                text = elem.get_text()
                if text.strip() and len(text.strip()) > 5:
                    parts.append(text.strip())

        return "\n\n".join(parts)

    def _extract_solutions(self, soup: BeautifulSoup) -> list[str]:
        """Extract all solutions from an AoPS wiki problem page."""
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return []

        solutions = []
        in_solution = False
        current_solution: list[str] = []
        solution_num = 0

        for elem in content.children:
            if not hasattr(elem, "name"):
                continue
            if elem.name in ("h2", "h3"):
                heading = elem.get_text()
                if "Solution" in heading:
                    if current_solution:
                        solutions.append("\n".join(current_solution).strip())
                    current_solution = []
                    in_solution = True
                    solution_num += 1
                elif in_solution:
                    # New non-solution section
                    if current_solution:
                        solutions.append("\n".join(current_solution).strip())
                    in_solution = False
                    current_solution = []
            elif in_solution and elem.name in ("p", "dl", "ul", "ol"):
                current_solution.append(elem.get_text())

        if current_solution:
            solutions.append("\n".join(current_solution).strip())

        return [s for s in solutions if len(s) > 30]

    def _extract_answer(self, soup: BeautifulSoup, answer_type: str) -> Optional[str]:
        """Extract numerical or multiple-choice answer."""
        if answer_type == "proof":
            return None

        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return None

        # Look for bolded answer choices or numbers
        for b in content.find_all("b"):
            text = b.get_text().strip()
            if answer_type == "multiple_choice" and re.match(r"^[A-E]$", text):
                return text
            if answer_type == "integer" and re.match(r"^\d{1,3}$", text):
                return text

        # Pattern: "The answer is X" in text
        text = content.get_text()
        m = re.search(r"answer\s+is\s+\(?([A-E0-9]+)\)?", text, re.IGNORECASE)
        if m:
            return m.group(1)

        return None

    async def _fetch_problem(
        self,
        session: aiohttp.ClientSession,
        competition: str,
        year: int,
        number: int,
        config: dict,
    ) -> Optional[OlympiadProblem]:
        """Fetch a single problem from AoPS wiki."""
        async with self._semaphore:
            template = config["wiki_template"]
            page = template.format(year=year, n=number)
            url = f"{AOPS_WIKI}/{page}"

            html = await self._fetch(session, url)
            if not html:
                return None

            soup = BeautifulSoup(html, "lxml")
            statement = self._extract_statement(soup)
            if not statement or len(statement) < 15:
                return None

            solutions = self._extract_solutions(soup)
            answer = self._extract_answer(soup, config["answer_type"])
            topics = _infer_topics(statement, solutions)
            difficulty = _difficulty(competition, number)

            return OlympiadProblem(
                problem_id=f"{competition}-{year}-{number:02d}",
                competition=competition,
                year=year,
                number=number,
                statement=statement,
                solutions=solutions[:5],  # Cap at 5 solutions
                answer=answer,
                answer_type=config["answer_type"],
                difficulty=difficulty,
                source_url=url,
                topics=topics,
            )

    async def _harvest_competition(self, session: aiohttp.ClientSession, competition: str) -> int:
        """Harvest all problems for a single competition."""
        config = COMPETITIONS[competition]
        output_file = self.output_dir / f"{competition.lower()}.jsonl"

        logger.info(
            f"Harvesting {competition}: "
            f"{len(config['years'])} years x {config['problems']} problems = "
            f"{len(config['years']) * config['problems']} total"
        )

        tasks = [
            self._fetch_problem(session, competition, year, n + 1, config)
            for year in config["years"]
            for n in range(config["problems"])
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        saved = 0
        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, OlympiadProblem):
                    await f.write(json.dumps(asdict(result)) + "\n")
                    self._stats["problems"] += 1
                    saved += 1
                elif isinstance(result, Exception):
                    self._stats["errors"] += 1

        logger.info(f"  {competition}: {saved} problems saved")
        return saved

    async def harvest_imo_official(self, session: aiohttp.ClientSession) -> int:
        """
        Harvest IMO problems from the official IMO website.

        The IMO official website has all problems in English, French, and other languages.
        URL pattern: https://www.imo-official.org/problems.aspx (index page)
        Each year's problems at: https://www.imo-official.org/year_info.aspx?year=YYYY
        """
        output_file = self.output_dir / "imo_official.jsonl"
        saved = 0

        for year in range(1959, 2025):
            url = f"{IMO_OFFICIAL}/year_info.aspx?year={year}"
            html = await self._fetch(session, url)
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")

            # Find problem links on the year page
            for problem_num in range(1, 7):
                # Problems listed in a table with links like problem.aspx?year=YYYY&problem=N
                prob_url = f"{IMO_OFFICIAL}/problem_shortlist.aspx?year={year}"
                problem_links = soup.find_all("a", href=re.compile(r"problem\.aspx"))

                for link in problem_links:
                    href = link.get("href", "")
                    if f"problem={problem_num}" in href:
                        full_url = f"{IMO_OFFICIAL}/{href}" if not href.startswith("http") else href

                        prob_html = await self._fetch(session, full_url)
                        if not prob_html:
                            continue

                        prob_soup = BeautifulSoup(prob_html, "lxml")
                        # Extract problem text from the official page
                        text_div = prob_soup.find("div", class_="problem_statement") or prob_soup.find("td", class_="problem_body")
                        if text_div:
                            statement = text_div.get_text(separator="\n").strip()
                            if len(statement) > 30:
                                record = {
                                    "problem_id": f"IMO_official-{year}-{problem_num:02d}",
                                    "competition": "IMO",
                                    "year": year,
                                    "number": problem_num,
                                    "statement": statement,
                                    "solutions": [],
                                    "answer": None,
                                    "answer_type": "proof",
                                    "difficulty": _difficulty("IMO", problem_num),
                                    "source_url": full_url,
                                    "topics": _infer_topics(statement, []),
                                    "source": "imo_official",
                                }
                                async with aiofiles.open(output_file, "a") as f:
                                    await f.write(json.dumps(record) + "\n")
                                saved += 1
                        break

        logger.info(f"IMO official: {saved} problems saved")
        return saved

    async def harvest_all(self) -> int:
        """Harvest all configured competitions."""
        async with aiohttp.ClientSession(
            headers={"User-Agent": "ProofCoach-Research/1.0 (calebnewtonusc@gmail.com)"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as session:
            total = 0
            for competition in self.competitions:
                if competition not in COMPETITIONS:
                    logger.warning(f"Unknown competition: {competition}")
                    continue
                n = await self._harvest_competition(session, competition)
                total += n

        logger.success(
            f"Olympiad harvest complete: {self._stats['problems']} problems, "
            f"{self._stats['errors']} errors"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Harvest competition math problems")
    parser.add_argument("--all", action="store_true", help="Harvest all competitions")
    parser.add_argument("--competition", nargs="+", default=None,
                        help=f"Competitions: {list(COMPETITIONS.keys())}")
    parser.add_argument("--output-dir", default="data/raw/olympiads")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--list", action="store_true", help="List available competitions")
    args = parser.parse_args()

    if args.list:
        for comp, cfg in COMPETITIONS.items():
            n_problems = len(cfg["years"]) * cfg["problems"]
            print(f"  {comp:<15} {n_problems:>6} problems  ({min(cfg['years'])}-{max(cfg['years'])})")
        raise SystemExit(0)

    competitions = list(COMPETITIONS.keys()) if args.all else args.competition

    harvester = OlympiadHarvester(
        output_dir=args.output_dir,
        workers=args.workers,
        competitions=competitions,
    )
    n = asyncio.run(harvester.harvest_all())
    print(f"\nTotal problems harvested: {n:,}")
