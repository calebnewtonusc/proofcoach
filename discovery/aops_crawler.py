"""
AoPS Crawler — Art of Problem Solving wiki + forum

Crawls:
  - aops.com/wiki — structured topic articles and solution pages per problem
  - aops.com/Forum — community solution posts (5-20 per problem)

Output: JSONL with problem, all community solutions, metadata
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

AOPS_BASE = "https://artofproblemsolving.com"
AOPS_WIKI = f"{AOPS_BASE}/wiki"
AOPS_FORUM = f"{AOPS_BASE}/Forum"

# Competition-specific wiki paths
COMPETITION_PATHS = {
    "AMC_8": "AMC_8_Problems_and_Solutions",
    "AMC_10": "AMC_10_Problems_and_Solutions",
    "AMC_12": "AMC_12_Problems_and_Solutions",
    "AIME": "AIME_Problems_and_Solutions",
    "USAMO": "USAMO_Problems_and_Solutions",
    "IMO": "IMO_Problems_and_Solutions",
    "HMMT": "HMMT_Problems_and_Solutions",
    "ARML": "ARML_Problems_and_Solutions",
}

TOPIC_ARTICLES = [
    "Pigeonhole_Principle",
    "AM-GM_Inequality",
    "Cauchy-Schwarz_Inequality",
    "Vieta's_Formulas",
    "Modular_arithmetic",
    "Fermat's_Little_Theorem",
    "Chinese_Remainder_Theorem",
    "Generating_functions",
    "Stars_and_Bars",
    "Inclusion-exclusion_principle",
    "Law_of_Cosines",
    "Power_of_a_Point",
    "Ptolemy's_Theorem",
    "Menelaus'_Theorem",
    "Ceva's_Theorem",
]


@dataclass
class AoPSSolution:
    """A single community solution post."""

    author: str
    content: str
    upvotes: int
    approach_tags: list[str]
    post_id: str


@dataclass
class AoPSProblem:
    """A competition problem with all community solutions."""

    problem_id: str
    competition: str
    year: int
    number: int
    statement: str
    answer: Optional[str]
    difficulty: Optional[int]
    topics: list[str]
    solutions: list[AoPSSolution]
    wiki_url: str
    forum_url: Optional[str]


class AoPSCrawler:
    """
    Async crawler for Art of Problem Solving wiki + forum.

    Usage:
        async with AoPSCrawler(output_dir="data/raw/aops", workers=20) as crawler:
            await crawler.crawl_all()
    """

    DELAY_MIN = 0.5
    DELAY_MAX = 1.5
    # RATE_LIMIT is created per-instance in __init__ so it is bound to the
    # running event loop (class-level asyncio.Semaphore raises RuntimeError
    # in Python 3.10+ because no event loop exists at class-definition time).

    def __init__(
        self,
        output_dir: Path | str,
        workers: int = 20,
        session_cookie: str = "",
        competitions: Optional[list[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.session_cookie = session_cookie
        self.competitions = competitions or list(COMPETITION_PATHS.keys())
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {"problems": 0, "solutions": 0, "errors": 0}
        self.RATE_LIMIT = asyncio.Semaphore(10)  # max 10 concurrent requests
        # Initialised once here so _crawl_competition does not reset it for
        # every competition and break concurrent tasks that already hold it.
        self._crawl_semaphore = asyncio.Semaphore(
            workers
        )  # per-competition page concurrency

    async def __aenter__(self) -> "AoPSCrawler":
        headers = {
            "User-Agent": "ProofCoach Research Crawler / calebnewtonusc@gmail.com",
            "Accept": "text/html,application/xhtml+xml",
        }
        if self.session_cookie:
            headers["Cookie"] = f"aops_session={self.session_cookie}"

        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
        logger.info(f"Crawl complete: {self._stats}")

    async def crawl_all(self) -> None:
        """Crawl all competitions and topic articles."""
        tasks = []

        # Crawl each competition
        for competition in self.competitions:
            tasks.append(self._crawl_competition(competition))

        # Crawl topic articles (convert to tutoring dialogues later)
        tasks.append(self._crawl_topic_articles())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _crawl_competition(self, competition: str) -> None:
        """Crawl all problems for a competition."""
        wiki_path = COMPETITION_PATHS.get(competition)
        if not wiki_path:
            logger.warning(f"Unknown competition: {competition}")
            return

        logger.info(f"Crawling {competition}...")
        output_file = self.output_dir / f"{competition.lower()}.jsonl"

        # Fetch the index page
        index_url = f"{AOPS_WIKI}/index.php/{wiki_path}"
        html = await self._fetch(index_url)
        if not html:
            return

        soup = BeautifulSoup(html, "lxml")
        problem_links = self._extract_problem_links(soup, competition)

        logger.info(f"{competition}: found {len(problem_links)} problem links")

        problems_to_fetch = []
        for url, year, number in problem_links:
            problems_to_fetch.append(
                self._fetch_problem_with_semaphore(url, competition, year, number)
            )

        results = await asyncio.gather(*problems_to_fetch, return_exceptions=True)

        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, AoPSProblem):
                    await f.write(json.dumps(asdict(result)) + "\n")
                    self._stats["problems"] += 1
                    self._stats["solutions"] += len(result.solutions)
                elif isinstance(result, Exception):
                    logger.debug(f"Error: {result}")
                    self._stats["errors"] += 1

    async def _fetch_problem_with_semaphore(
        self, url: str, competition: str, year: int, number: int
    ) -> Optional[AoPSProblem]:
        """Fetch a single problem page under the crawl semaphore."""
        async with self._crawl_semaphore:
            return await self._fetch_problem_page(url, competition, year, number)

    def _extract_problem_links(
        self, soup: BeautifulSoup, competition: str
    ) -> list[tuple[str, int, int]]:
        """Extract (url, year, problem_number) tuples from an index page."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Match patterns like /wiki/2023_AMC_12A_Problems/Problem_15
            match = re.search(
                r"/wiki/(\d{4})_"
                + re.escape(competition.replace("_", r"[_\s]"))
                + r"[_A-Z]*/Problem[_/](\d+)",
                href,
                re.IGNORECASE,
            )
            if match:
                year = int(match.group(1))
                number = int(match.group(2))
                full_url = urljoin(AOPS_BASE, href)
                links.append((full_url, year, number))

        return list(set(links))  # deduplicate

    async def _fetch_problem_page(
        self, url: str, competition: str, year: int, number: int
    ) -> Optional[AoPSProblem]:
        """Fetch and parse a single problem page."""
        html = await self._fetch(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        # Extract problem statement
        statement = self._extract_problem_statement(soup)
        if not statement:
            return None

        # Extract answer if present
        answer = self._extract_answer(soup)

        # Extract solutions from wiki page
        wiki_solutions = self._extract_wiki_solutions(soup)

        # Try to find the forum thread
        forum_url = self._find_forum_url(soup)
        forum_solutions = []
        if forum_url:
            forum_solutions = await self._fetch_forum_solutions(forum_url)

        all_solutions = wiki_solutions + forum_solutions
        if not all_solutions:
            return None

        # Infer topics from content
        topics = self._infer_topics(statement, all_solutions)
        difficulty = self._estimate_difficulty(competition, number)

        problem_id = f"{competition}-{year}-{number:02d}"

        return AoPSProblem(
            problem_id=problem_id,
            competition=competition,
            year=year,
            number=number,
            statement=statement,
            answer=answer,
            difficulty=difficulty,
            topics=topics,
            solutions=all_solutions,
            wiki_url=url,
            forum_url=forum_url,
        )

    def _extract_problem_statement(self, soup: BeautifulSoup) -> str:
        """Extract the problem statement from the wiki page."""
        # AoPS wiki typically has the problem in the first div.mw-parser-output p
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            return ""

        # Problem statement is usually before the first "Solution" heading
        statement_parts = []
        for element in content_div.children:
            if hasattr(element, "name"):
                if element.name in ("h2", "h3") and "Solution" in element.get_text():
                    break
                if element.name == "p":
                    text = element.get_text()
                    if text.strip():
                        statement_parts.append(text.strip())

        return "\n\n".join(statement_parts)

    def _extract_answer(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the answer (for AMC multiple choice or AIME integer)."""
        # Look for answer boxes
        for span in soup.find_all("span", class_="mw-headline"):
            if "Answer" in span.get_text():
                parent = span.parent
                if parent and parent.next_sibling:
                    return parent.next_sibling.get_text().strip()
        return None

    def _extract_wiki_solutions(self, soup: BeautifulSoup) -> list[AoPSSolution]:
        """Extract solutions embedded in the wiki page."""
        solutions = []
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            return solutions

        # Find all Solution sections
        in_solution = False
        solution_text_parts: list[str] = []
        solution_number = 0

        for element in content_div.children:
            if not hasattr(element, "name"):
                continue

            if element.name in ("h2", "h3"):
                heading = element.get_text()
                if "Solution" in heading:
                    if in_solution and solution_text_parts:
                        solutions.append(
                            AoPSSolution(
                                author="aops_wiki",
                                content="\n".join(solution_text_parts).strip(),
                                upvotes=0,
                                approach_tags=self._infer_approach_tags(
                                    "\n".join(solution_text_parts)
                                ),
                                post_id=f"wiki_{solution_number}",
                            )
                        )
                        solution_text_parts = []
                    in_solution = True
                    solution_number += 1
                elif in_solution:
                    # New non-solution section, stop
                    if solution_text_parts:
                        solutions.append(
                            AoPSSolution(
                                author="aops_wiki",
                                content="\n".join(solution_text_parts).strip(),
                                upvotes=0,
                                approach_tags=self._infer_approach_tags(
                                    "\n".join(solution_text_parts)
                                ),
                                post_id=f"wiki_{solution_number}",
                            )
                        )
                    in_solution = False
                    solution_text_parts = []

            elif in_solution and element.name == "p":
                solution_text_parts.append(element.get_text())

        return solutions

    def _find_forum_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the AoPS forum thread for this problem."""
        for a in soup.find_all("a", href=True):
            if "Forum/viewtopic" in a["href"] or "/community/" in a["href"]:
                return urljoin(AOPS_BASE, a["href"])
        return None

    async def _fetch_forum_solutions(self, forum_url: str) -> list[AoPSSolution]:
        """Fetch solution posts from the forum thread."""
        html = await self._fetch(forum_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        solutions = []

        for post_div in soup.find_all("div", class_="postbody"):
            author_div = post_div.find("div", class_="poster")
            content_div = post_div.find("div", class_="posttext")

            if not content_div:
                continue

            author = author_div.get_text().strip() if author_div else "anonymous"
            content = content_div.get_text().strip()

            # Skip very short posts (likely comments, not solutions)
            if len(content) < 100:
                continue

            # Skip posts that don't contain math
            if not re.search(r"\$.*?\$|\\[a-z]+{", content):
                continue

            # Look for vote count
            # PC-19: Guard against ValueError/AttributeError when the vote-count
            # span contains non-integer text (e.g. "—" or empty string).
            votes_span = post_div.find("span", class_="vote-count")
            upvotes = 0
            if votes_span:
                try:
                    upvotes = int(votes_span.get_text().strip())
                except (ValueError, AttributeError):
                    upvotes = 0

            post_id = post_div.get("id", f"post_{len(solutions)}")

            solutions.append(
                AoPSSolution(
                    author=self._anonymize(author),
                    content=content,
                    upvotes=upvotes,
                    approach_tags=self._infer_approach_tags(content),
                    post_id=post_id,
                )
            )

        # Sort by upvotes descending — higher quality first
        solutions.sort(key=lambda s: s.upvotes, reverse=True)

        # Cap at 10 solutions to avoid bloat
        return solutions[:10]

    async def _crawl_topic_articles(self) -> None:
        """Crawl topic knowledge articles from AoPS wiki."""
        output_file = self.output_dir / "topic_articles.jsonl"
        logger.info("Crawling topic articles...")

        async with aiofiles.open(output_file, "w") as f:
            for article in TOPIC_ARTICLES:
                url = f"{AOPS_WIKI}/index.php/{article}"
                html = await self._fetch(url)
                if not html:
                    continue

                soup = BeautifulSoup(html, "lxml")
                content_div = soup.find("div", class_="mw-parser-output")
                if not content_div:
                    continue

                content = content_div.get_text()
                record = {
                    "topic": article.replace("_", " "),
                    "url": url,
                    "content": content,
                    "type": "topic_article",
                }
                await f.write(json.dumps(record) + "\n")
                logger.debug(f"Topic article: {article}")

    def _infer_topics(self, statement: str, solutions: list[AoPSSolution]) -> list[str]:
        """Infer mathematical topics from problem content."""
        all_text = statement + " ".join(s.content for s in solutions)
        all_text_lower = all_text.lower()

        topic_keywords = {
            "number_theory": [
                "prime",
                "divisib",
                "modular",
                "gcd",
                "lcm",
                "congruent",
                "remainder",
                "fermat",
            ],
            "combinatorics": [
                "count",
                "combinat",
                "permut",
                "choose",
                "arrange",
                "pigeonhole",
                "generating function",
            ],
            "algebra": [
                "polynomial",
                "equation",
                "inequality",
                "sequence",
                "function",
                "factor",
                "vieta",
            ],
            "geometry": [
                "triangle",
                "circle",
                "angle",
                "perpendicular",
                "parallel",
                "chord",
                "tangent",
                "area",
            ],
            "probability": [
                "probabilit",
                "expected",
                "random",
                "event",
                "sample space",
            ],
            "calculus": [
                "limit",
                "derivative",
                "integral",
                "continuous",
                "differentiable",
            ],
        }

        topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text_lower for kw in keywords):
                topics.append(topic)

        return topics or ["general"]

    def _infer_approach_tags(self, solution_text: str) -> list[str]:
        """Infer the mathematical approach from solution text."""
        text_lower = solution_text.lower()
        tags = []

        approach_patterns = {
            "induction": ["induction", "base case", "inductive step"],
            "contradiction": [
                "contradiction",
                "assume for contradiction",
                "suppose not",
            ],
            "casework": ["case 1", "case 2", "cases:", "split into"],
            "constructive": ["construct", "let us build", "we build"],
            "pigeonhole": ["pigeonhole", "by the pigeonhole"],
            "modular_arithmetic": ["mod ", "modulo", "congruent to"],
            "vietas": ["vieta", "sum of roots", "product of roots"],
            "am_gm": ["am-gm", "am ≥ gm", "arithmetic mean"],
            "cauchy_schwarz": ["cauchy-schwarz", "cauchy schwarz"],
        }

        for tag, keywords in approach_patterns.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)

        return tags

    def _estimate_difficulty(self, competition: str, number: int) -> int:
        """Estimate problem difficulty on 1-10 scale."""
        if competition == "AMC_8":
            return min(1 + number // 4, 5)
        elif competition in ("AMC_10", "AMC_12"):
            return min(1 + number // 4, 7)
        elif competition == "AIME":
            return min(4 + number // 3, 9)
        elif competition in ("USAMO", "IMO"):
            # IMO problems 1,4 are easiest, 3,6 are hardest
            if number in (1, 4):
                return 7
            elif number in (2, 5):
                return 9
            else:
                return 10
        return 5

    def _anonymize(self, username: str) -> str:
        """Anonymize a username for privacy."""
        import hashlib

        return "user_" + hashlib.md5(username.encode(), usedforsecurity=False).hexdigest()[:8]

    async def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with rate limiting and retries."""
        async with self.RATE_LIMIT:
            for attempt in range(3):
                try:
                    await asyncio.sleep(
                        self.DELAY_MIN + (self.DELAY_MAX - self.DELAY_MIN) * 0.5
                    )
                    if self._session is None:
                        raise RuntimeError(
                            "AoPSCrawler._fetch called before session was initialised. "
                            "Use 'async with AoPSCrawler(...) as crawler:' to ensure "
                            "the session is created via __aenter__."
                        )
                    async with self._session.get(url) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status == 429:
                            wait = 2**attempt * 5
                            logger.warning(f"Rate limited. Waiting {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            logger.debug(f"HTTP {response.status} for {url}")
                            return None
                except Exception as e:
                    if attempt == 2:
                        logger.debug(f"Failed to fetch {url}: {e}")
                    await asyncio.sleep(2**attempt)
        return None


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        session_cookie = os.getenv("AOPS_SESSION_COOKIE", "")
        async with AoPSCrawler(
            output_dir="data/raw/aops",
            workers=20,
            session_cookie=session_cookie,
        ) as crawler:
            await crawler.crawl_all()

    asyncio.run(main())
