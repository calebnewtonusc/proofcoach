"""
lean4_mathlib.py — GitHub API crawler for Lean 4 Mathlib theorems + proofs.

Downloads the full mathlib4 repository file tree from GitHub API and extracts
(theorem_statement, informal_description, lean4_proof) triples.

Target: 50k+ theorem-proof pairs from:
  - leanprover-community/mathlib4 (primary: 50k+ theorems)
  - leanprover/lean4 (core library)
  - leanprover-community/batteries (standard library extensions)

Usage:
    export GITHUB_TOKEN=ghp_...
    python discovery/lean4_mathlib.py --output-dir data/raw/lean4
    python discovery/lean4_mathlib.py --repo batteries --output-dir data/raw/lean4
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"

TARGET_REPOS = [
    {"owner": "leanprover-community", "repo": "mathlib4", "branch": "master"},
    {"owner": "leanprover", "repo": "lean4", "branch": "master"},
    {"owner": "leanprover-community", "repo": "batteries", "branch": "main"},
]

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "lean4"


@dataclass
class Lean4Theorem:
    """A single theorem-proof pair extracted from a .lean file."""

    theorem_id: str
    repo: str
    file_path: str
    theorem_name: str
    namespace: str
    statement: str
    proof: str
    proof_type: str  # "tactic" | "term" | "decide" | "simp" | "other"
    docstring: Optional[str]
    line_number: int
    tags: list[str]


def extract_theorems_from_lean(
    content: str, file_path: str, repo: str
) -> list[Lean4Theorem]:
    """
    Parse a Lean 4 source file and extract all theorem-proof pairs.

    Handles:
      - theorem / lemma / proposition declarations
      - with and without tactic proofs (by ... blocks)
      - term-mode proofs (:= expr)
      - docstrings preceding theorems
    """
    theorems = []
    lines = content.splitlines()
    n = len(lines)

    # Track current namespace
    namespace_stack: list[str] = []

    # Regex patterns
    theorem_decl_re = re.compile(
        r"^(?:@\[.*?\]\s*)*"  # optional attributes
        r"(?:protected\s+|private\s+|public\s+)?"  # optional visibility
        r"(?:theorem|lemma|proposition)\s+"
        r"(\w+)"  # name
        r"([^:=]*?)"  # params (lazy)
        r"\s*:\s*"  # colon
        r"(.+?)$",  # type (rest of line, may continue)
        re.MULTILINE,
    )
    namespace_re = re.compile(r"^namespace\s+(\w+)", re.MULTILINE)
    end_re = re.compile(r"^end\s+(\w+)?", re.MULTILINE)

    # Walk through lines tracking namespaces and extracting theorems
    i = 0
    while i < n:
        line = lines[i]

        # Track namespaces
        ns_m = namespace_re.match(line.strip())
        if ns_m:
            namespace_stack.append(ns_m.group(1))
            i += 1
            continue

        end_m = end_re.match(line.strip())
        if end_m and namespace_stack:
            namespace_stack.pop()
            i += 1
            continue

        # Collect docstring
        docstring = None
        if line.strip().startswith("/-!") or line.strip().startswith("/--"):
            doc_lines = [line]
            j = i + 1
            while j < n and "-/" not in lines[j]:
                doc_lines.append(lines[j])
                j += 1
            if j < n:
                doc_lines.append(lines[j])
            docstring = "\n".join(doc_lines).strip()
            # Don't advance i — let the theorem decl be picked up below
            i = j + 1
            continue

        # Detect theorem declaration
        decl_m = theorem_decl_re.match(line.strip())
        if decl_m:
            theorem_name = decl_m.group(1)
            decl_m.group(2).strip()
            decl_m.group(3).strip()

            # Collect multi-line statement (until := or by or where)
            statement_lines = [line.strip()]
            j = i + 1
            while j < n:
                next_line = lines[j].strip()
                statement_lines.append(next_line)
                if (
                    ":=" in next_line
                    or next_line.startswith("by ")
                    or next_line == "by"
                ):
                    break
                if next_line.startswith("theorem ") or next_line.startswith("lemma "):
                    j -= 1
                    break
                j += 1
                if j - i > 20:  # Cap statement collection
                    break

            full_statement = " ".join(statement_lines).strip()

            # Extract proof
            proof_lines = []
            proof_type = "other"
            k = j
            if k < n:
                proof_line = lines[k].strip()
                if ":=" in proof_line:
                    proof_type = "term"
                    # Collect term proof (balanced brackets)
                    proof_lines = [proof_line]
                    depth = proof_line.count("(") - proof_line.count(")")
                    m = k + 1
                    while m < n and (depth > 0 or (not lines[m].strip() and m - k < 3)):
                        proof_lines.append(lines[m].strip())
                        depth += lines[m].count("(") - lines[m].count(")")
                        m += 1
                        if m - k > 30:
                            break
                elif "by" in proof_line:
                    proof_type = "tactic"
                    # Collect tactic block (indented lines)
                    proof_lines = [proof_line]
                    base_indent = len(lines[k]) - len(lines[k].lstrip())
                    m = k + 1
                    while m < n:
                        if lines[m].strip() == "":
                            proof_lines.append("")
                            m += 1
                            continue
                        current_indent = len(lines[m]) - len(lines[m].lstrip())
                        if (
                            current_indent <= base_indent
                            and lines[m].strip()
                            and not lines[m].strip().startswith("--")
                        ):
                            break
                        proof_lines.append(lines[m])
                        m += 1
                        if m - k > 50:
                            break

            # Detect common proof types
            proof_text = "\n".join(proof_lines).strip()
            if "simp" in proof_text and proof_type == "tactic":
                proof_type = "simp"
            elif "decide" in proof_text:
                proof_type = "decide"
            elif (
                "omega" in proof_text
                or "ring" in proof_text
                or "linarith" in proof_text
            ):
                proof_type = "tactic"

            # Skip very short statements (likely forward declarations)
            if len(full_statement) < 20:
                i += 1
                continue

            namespace = ".".join(namespace_stack)
            theorem_id = f"{repo}/{file_path}:{i + 1}:{theorem_name}"

            # Infer math tags from statement
            tags = _infer_lean4_tags(full_statement, proof_text)

            theorems.append(
                Lean4Theorem(
                    theorem_id=theorem_id,
                    repo=repo,
                    file_path=file_path,
                    theorem_name=theorem_name,
                    namespace=namespace,
                    statement=full_statement[:4000],
                    proof=proof_text[:4000],
                    proof_type=proof_type,
                    docstring=docstring,
                    line_number=i + 1,
                    tags=tags,
                )
            )
            i = max(j, k) + 1
            continue

        i += 1

    return theorems


def _infer_lean4_tags(statement: str, proof: str) -> list[str]:
    """Infer mathematical topic tags from Lean 4 theorem content."""
    text = (statement + " " + proof).lower()
    tags = []

    topic_patterns = {
        "algebra": ["ring", "field", "group", "monoid", "module", "ideal", "algebra"],
        "number_theory": [
            "nat",
            "int",
            "prime",
            "divisible",
            "gcd",
            "lcm",
            "modular",
            "zmod",
        ],
        "topology": ["open", "closed", "continuous", "metric", "topological"],
        "analysis": [
            "real",
            "complex",
            "limit",
            "deriv",
            "integral",
            "convergence",
            "series",
        ],
        "combinatorics": ["finset", "multiset", "permutation", "count", "card"],
        "logic": ["iff", "ite", "classical", "decidable", "propositional"],
        "linear_algebra": [
            "matrix",
            "vector",
            "linear",
            "span",
            "basis",
            "determinant",
        ],
        "category_theory": ["functor", "category", "morphism", "adjunction", "natural"],
        "order_theory": ["lattice", "poset", "orderiso", "completelattice"],
        "measure_theory": ["measure", "integral", "probability", "ae"],
    }

    for tag, keywords in topic_patterns.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    return tags or ["general"]


class Lean4MathlibCrawler:
    """
    Async crawler for Lean 4 Mathlib4 theorem-proof pairs.

    Fetches the full repository file tree using the GitHub Trees API,
    then downloads each .lean file and extracts theorems.
    """

    REQUEST_DELAY = 0.1  # seconds between requests

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        token: Optional[str] = None,
        workers: int = 20,
        repos: Optional[list[dict]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.workers = workers
        self.repos = repos or TARGET_REPOS
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"files": 0, "theorems": 0, "errors": 0}

    def _headers(self) -> dict:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _fetch_json(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[dict]:
        """Fetch GitHub API JSON with rate-limit handling."""
        for attempt in range(5):
            await asyncio.sleep(self.REQUEST_DELAY)
            try:
                async with session.get(
                    url,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status in (403, 429):
                        remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
                        if remaining < 10:
                            reset = int(
                                resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                            )
                            wait = max(1, reset - int(time.time())) + 5
                            logger.warning(f"Rate limited. Waiting {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            await asyncio.sleep(2**attempt)
                    elif resp.status == 404:
                        return None
                    else:
                        await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.debug(f"Fetch error ({url}): {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _fetch_raw(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        """Fetch raw file content."""
        for attempt in range(3):
            await asyncio.sleep(0.05)
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    elif resp.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        return None
            except Exception as e:
                logger.debug(f"Raw fetch error: {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _get_file_tree(
        self, session: aiohttp.ClientSession, owner: str, repo: str, branch: str
    ) -> list[dict]:
        """Fetch the full recursive file tree for a repo."""
        url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        data = await self._fetch_json(session, url)
        if not data:
            logger.warning(f"Could not fetch tree for {owner}/{repo}")
            return []
        tree = data.get("tree", [])
        lean_files = [
            f
            for f in tree
            if f.get("type") == "blob"
            and f.get("path", "").endswith(".lean")
            and ".lake/" not in f.get("path", "")
        ]
        logger.info(f"  {owner}/{repo}: {len(lean_files)} .lean files in tree")
        return lean_files

    async def _process_file(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        branch: str,
        file_info: dict,
        output_file: Path,
    ) -> int:
        """Download and parse a single .lean file, append theorems to output_file."""
        async with self._semaphore:
            path = file_info["path"]
            raw_url = f"{GITHUB_RAW}/{owner}/{repo}/{branch}/{path}"

            content = await self._fetch_raw(session, raw_url)
            if not content:
                return 0

            try:
                theorems = extract_theorems_from_lean(content, path, f"{owner}/{repo}")
            except Exception as e:
                logger.debug(f"Parse error for {path}: {e}")
                self._stats["errors"] += 1
                return 0

            if not theorems:
                return 0

            async with aiofiles.open(output_file, "a") as f:
                for thm in theorems:
                    await f.write(json.dumps(asdict(thm)) + "\n")

            self._stats["files"] += 1
            self._stats["theorems"] += len(theorems)
            return len(theorems)

    async def crawl_repo(
        self, session: aiohttp.ClientSession, repo_config: dict
    ) -> int:
        """Crawl a single repository."""
        owner = repo_config["owner"]
        repo = repo_config["repo"]
        branch = repo_config.get("branch", "master")

        logger.info(f"Crawling {owner}/{repo} (branch={branch})...")
        output_file = self.output_dir / f"{repo.replace('/', '_')}.jsonl"

        # Clear existing output
        if output_file.exists():
            output_file.unlink()

        # Get file tree
        lean_files = await self._get_file_tree(session, owner, repo, branch)
        if not lean_files:
            return 0

        # Process all files in parallel
        tasks = [
            self._process_file(session, owner, repo, branch, f, output_file)
            for f in lean_files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = sum(r for r in results if isinstance(r, int))

        logger.info(f"  {owner}/{repo}: {total} theorems extracted")
        return total

    async def crawl_all(self) -> int:
        """Crawl all configured repositories."""
        async with aiohttp.ClientSession(
            headers={"User-Agent": "ProofCoach-Research/1.0"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as session:
            total = 0
            for repo_config in self.repos:
                n = await self.crawl_repo(session, repo_config)
                total += n

        logger.success(
            f"Lean4 crawl complete: {self._stats['files']} files, "
            f"{self._stats['theorems']} theorems, {self._stats['errors']} errors"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Crawl Lean 4 Mathlib for theorem-proof pairs"
    )
    parser.add_argument("--output-dir", default="data/raw/lean4")
    parser.add_argument(
        "--repo", nargs="+", help="Specific repos: mathlib4 lean4 batteries"
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    repos = None
    if args.repo:
        name_to_config = {r["repo"]: r for r in TARGET_REPOS}
        repos = [name_to_config[r] for r in args.repo if r in name_to_config]

    crawler = Lean4MathlibCrawler(
        output_dir=args.output_dir,
        workers=args.workers,
        repos=repos,
    )
    n = asyncio.run(crawler.crawl_all())
    print(f"\nTotal theorems extracted: {n:,}")
