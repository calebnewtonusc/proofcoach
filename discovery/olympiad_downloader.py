"""
Olympiad Downloader — AMC/AIME/USAMO/IMO archives

Downloads problem sets from:
  - Art of Problem Solving wiki (primary source for AMC/AIME/USAMO)
  - IMO official website (imo-official.org)
  - HMMT, ARML, Putnam (via AoPS wiki)

Output: JSONL with problem statement, official solution, year, competition, number
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

IMO_OFFICIAL = "https://www.imo-official.org"
AOPS_WIKI = "https://artofproblemsolving.com/wiki/index.php"


@dataclass
class OlympiadProblem:
    """A single competition problem with official solution."""
    problem_id: str
    competition: str
    year: int
    number: int
    statement: str
    official_solution: Optional[str]
    answer: Optional[str]          # For AMC/AIME with numerical answers
    answer_type: str               # "multiple_choice", "integer", "proof"
    difficulty_estimate: int       # 1-10
    source_url: str


COMPETITION_CONFIG = {
    # (url_pattern, years, problem_count, answer_type)
    "AMC_8": {
        "years": range(1999, 2026),
        "problems": 25,
        "answer_type": "multiple_choice",
    },
    "AMC_10A": {
        "years": range(2002, 2026),
        "problems": 30,
        "answer_type": "multiple_choice",
    },
    "AMC_10B": {
        "years": range(2002, 2026),
        "problems": 30,
        "answer_type": "multiple_choice",
    },
    "AMC_12A": {
        "years": range(2002, 2026),
        "problems": 30,
        "answer_type": "multiple_choice",
    },
    "AMC_12B": {
        "years": range(2002, 2026),
        "problems": 30,
        "answer_type": "multiple_choice",
    },
    "AIME_I": {
        "years": range(1983, 2026),
        "problems": 15,
        "answer_type": "integer",
    },
    "AIME_II": {
        "years": range(2000, 2026),
        "problems": 15,
        "answer_type": "integer",
    },
    "USAMO": {
        "years": range(1972, 2026),
        "problems": 6,
        "answer_type": "proof",
    },
    "IMO": {
        "years": range(1959, 2026),
        "problems": 6,
        "answer_type": "proof",
    },
}


class OlympiadDownloader:
    """Downloads competition math problems from AoPS wiki and official sources."""

    def __init__(
        self,
        output_dir: Path | str,
        competitions: Optional[list[str]] = None,
        workers: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.competitions = competitions or list(COMPETITION_CONFIG.keys())
        self.workers = workers
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"problems": 0, "errors": 0}
        self._session: Optional[aiohttp.ClientSession] = None

    async def download_all_competitions(self) -> None:
        """Download all configured competitions."""
        session_config = aiohttp.ClientSession(
            headers={"User-Agent": "ProofCoach Research / calebnewtonusc"},
            timeout=aiohttp.ClientTimeout(total=30),
        )

        async with session_config as session:
            self._session = session
            tasks = [self._download_competition(comp) for comp in self.competitions]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Download complete: {self._stats}")

    async def _download_competition(self, competition: str) -> None:
        """Download all problems for a single competition."""
        config = COMPETITION_CONFIG.get(competition)
        if not config:
            logger.warning(f"Unknown competition: {competition}")
            return

        output_file = self.output_dir / f"{competition.lower()}.jsonl"
        logger.info(f"Downloading {competition} ({len(config['years'])} years × {config['problems']} problems)...")

        tasks = []
        for year in config["years"]:
            for number in range(1, config["problems"] + 1):
                tasks.append(
                    self._download_problem(competition, year, number, config["answer_type"])
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, OlympiadProblem):
                    await f.write(json.dumps(asdict(result)) + "\n")
                    self._stats["problems"] += 1
                elif isinstance(result, Exception):
                    self._stats["errors"] += 1

        logger.info(f"{competition}: done, {self._stats['problems']} problems total")

    async def _download_problem(
        self, competition: str, year: int, number: int, answer_type: str
    ) -> Optional[OlympiadProblem]:
        """Download a single problem from AoPS wiki."""
        async with self._semaphore:
            url = self._build_wiki_url(competition, year, number)
            html = await self._fetch(url)
            if not html:
                return None

            soup = BeautifulSoup(html, "lxml")
            statement = self._extract_statement(soup)
            if not statement or len(statement) < 20:
                return None

            official_solution = self._extract_official_solution(soup)
            answer = self._extract_answer(soup, answer_type)
            difficulty = self._estimate_difficulty(competition, number)

            return OlympiadProblem(
                problem_id=f"{competition}-{year}-{number:02d}",
                competition=competition,
                year=year,
                number=number,
                statement=statement,
                official_solution=official_solution,
                answer=answer,
                answer_type=answer_type,
                difficulty_estimate=difficulty,
                source_url=url,
            )

    def _build_wiki_url(self, competition: str, year: int, number: int) -> str:
        """Build the AoPS wiki URL for a problem."""
        # PC-18: The original replace("_", "_") was a no-op. The variable
        # comp_clean was also unused. Remove both dead lines.
        # comp_clean is not referenced below — the per-competition if-branches
        # use the original `competition` string directly.

        if competition.startswith("AMC_8"):
            page = f"{year}_AMC_8_Problems/Problem_{number}"
        elif competition.startswith("AMC_10A"):
            page = f"{year}_AMC_10A_Problems/Problem_{number}"
        elif competition.startswith("AMC_10B"):
            page = f"{year}_AMC_10B_Problems/Problem_{number}"
        elif competition.startswith("AMC_12A"):
            page = f"{year}_AMC_12A_Problems/Problem_{number}"
        elif competition.startswith("AMC_12B"):
            page = f"{year}_AMC_12B_Problems/Problem_{number}"
        elif competition.startswith("AIME_I"):
            page = f"{year}_AIME_I_Problems/Problem_{number}"
        elif competition.startswith("AIME_II"):
            page = f"{year}_AIME_II_Problems/Problem_{number}"
        elif competition == "USAMO":
            page = f"{year}_USAMO_Problems/Problem_{number}"
        elif competition == "IMO":
            page = f"{year}_IMO_Problems/Problem_{number}"
        else:
            page = f"{year}_{competition}_Problems/Problem_{number}"

        return f"{AOPS_WIKI}/{page}"

    def _extract_statement(self, soup: BeautifulSoup) -> str:
        """Extract problem statement from wiki page."""
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return ""

        parts = []
        for elem in content.children:
            if not hasattr(elem, "name"):
                continue
            if elem.name in ("h2", "h3"):
                break  # Stop at first heading (Solution, See Also, etc.)
            if elem.name in ("p", "dl", "ul", "ol"):
                text = elem.get_text()
                if text.strip():
                    parts.append(text.strip())

        return "\n\n".join(parts)

    def _extract_official_solution(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the first solution from the wiki page."""
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return None

        in_solution = False
        parts = []
        for elem in content.children:
            if not hasattr(elem, "name"):
                continue
            if elem.name in ("h2", "h3"):
                heading = elem.get_text()
                if "Solution" in heading and not in_solution:
                    in_solution = True
                elif in_solution and "Solution" not in heading:
                    break  # Next section
            elif in_solution and elem.name in ("p", "dl", "ul", "ol"):
                text = elem.get_text()
                if text.strip():
                    parts.append(text.strip())
                if len(parts) > 20:
                    break  # Don't grab too much

        return "\n\n".join(parts) if parts else None

    def _extract_answer(self, soup: BeautifulSoup, answer_type: str) -> Optional[str]:
        """Extract the answer from the page."""
        if answer_type == "proof":
            return None

        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return None

        # Look for bold text containing the answer
        for b in content.find_all("b"):
            text = b.get_text().strip()
            if answer_type == "multiple_choice" and re.match(r"^[A-E]$", text):
                return text
            elif answer_type == "integer" and re.match(r"^\d{1,3}$", text):
                return text

        return None

    def _estimate_difficulty(self, competition: str, number: int) -> int:
        """Estimate problem difficulty 1-10."""
        if "AMC_8" in competition:
            return max(1, min(5, 1 + number // 5))
        elif "AMC_10" in competition or "AMC_12" in competition:
            return max(1, min(7, 1 + number // 5))
        elif "AIME" in competition:
            return max(4, min(9, 3 + number // 2))
        elif competition in ("USAMO", "IMO"):
            if number in (1, 4):
                return 7
            elif number in (2, 5):
                return 9
            else:
                return 10
        return 5

    async def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with retry logic."""
        if self._session is None:
            raise RuntimeError(
                "OlympiadDownloader._fetch called before session was initialised. "
                "Call download_all_competitions() which sets up the session."
            )
        for attempt in range(3):
            try:
                await asyncio.sleep(0.2)  # polite delay
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 404:
                        return None  # Problem doesn't exist for this year
                    elif resp.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        return None
            except Exception as e:
                if attempt == 2:
                    logger.debug(f"Fetch failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None


class PutnamDownloader:
    """Downloads Putnam competition problems (1938-2024)."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def download_all(self) -> None:
        """Download all Putnam problems from AoPS wiki."""
        output_file = self.output_dir / "putnam.jsonl"
        logger.info("Downloading Putnam archives (1938-2024)...")

        async with aiohttp.ClientSession(
            headers={"User-Agent": "ProofCoach Research"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            self._session = session
            problems = []

            for year in range(1938, 2025):
                for section in ("A", "B"):
                    for number in range(1, 7):
                        page = f"{year}_Putnam_{section}{number}"
                        url = f"{AOPS_WIKI}/{page}"
                        html = await self._fetch(url)
                        if html:
                            soup = BeautifulSoup(html, "lxml")
                            content = soup.find("div", class_="mw-parser-output")
                            if content:
                                statement = content.get_text()[:2000]
                                problems.append({
                                    "problem_id": f"Putnam-{year}-{section}{number}",
                                    "competition": "Putnam",
                                    "year": year,
                                    "section": section,
                                    "number": number,
                                    "statement": statement,
                                    "answer_type": "proof",
                                    "difficulty_estimate": 7 + (number > 4),
                                    "source_url": url,
                                })

            async with aiofiles.open(output_file, "w") as f:
                for p in problems:
                    await f.write(json.dumps(p) + "\n")

        logger.info(f"Putnam: {len(problems)} problems downloaded")

    async def _fetch(self, url: str) -> Optional[str]:
        # PC-26: Log exceptions at debug level instead of silently swallowing
        # them. Silent failures make it impossible to diagnose crawl issues.
        try:
            await asyncio.sleep(0.15)
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception as exc:
            logger.debug(f"Putnam fetch failed for {url}: {exc}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download olympiad problem archives")
    parser.add_argument("--output-dir", default="data/raw/olympiads")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--putnam", action="store_true", help="Download Putnam archives")
    parser.add_argument("--competition", nargs="+", default=None,
                        help="Specific competitions to download")
    args = parser.parse_args()

    async def main():
        if args.putnam:
            downloader = PutnamDownloader(output_dir="data/raw/putnam")
            await downloader.download_all()
        else:
            downloader = OlympiadDownloader(
                output_dir=args.output_dir,
                workers=args.workers,
                competitions=args.competition,
            )
            await downloader.download_all_competitions()

    asyncio.run(main())
