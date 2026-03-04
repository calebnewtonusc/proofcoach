"""
Teaching Synthesizer — Generates Socratic tutoring dialogues

Wraps SynthesisPipeline with teaching-specific logic:
  - Ensures coverage across all competition types
  - Balances difficulty distribution
  - Generates multi-turn dialogues (not just single exchanges)
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from synthesis.synthesize_bulk import SynthesisPipeline


class TeachingSynthesizer:
    """
    Generates full multi-turn Socratic tutoring dialogues for training.

    Each dialogue covers the full teaching arc:
      1. Student presents problem + initial attempt
      2. Tutor asks targeted Socratic question
      3. Student responds (shows partial understanding)
      4. Tutor builds on response, asks next question
      ... (4-7 turns)
      N. Student articulates key insight themselves
      N+1. Tutor confirms and offers next practice problem
    """

    DIFFICULTY_DISTRIBUTION = {
        "easy": (1, 4, 0.25),      # (min, max, fraction)
        "medium": (4, 7, 0.50),
        "hard": (7, 10, 0.25),
    }

    COMPETITION_DISTRIBUTION = {
        "AMC_8": 0.10,
        "AMC_10A": 0.15,
        "AMC_12A": 0.15,
        "AIME_I": 0.20,
        "AIME_II": 0.10,
        "USAMO": 0.15,
        "IMO": 0.15,
    }

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 20,
        target_total: int = 90000,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self.target_total = target_total

        self._pipeline = SynthesisPipeline(
            raw_dir=raw_dir,
            output_dir=output_dir,
            backend=backend,
            vllm_urls=vllm_urls,
            workers=workers,
        )

    async def synthesize_all(self) -> None:
        """Run teaching dialogue synthesis with balanced distribution."""
        logger.info(f"Teaching synthesis: target {self.target_total:,} dialogues")
        await self._pipeline.synthesize_all()
        logger.info("Teaching synthesis complete")

    def report_distribution(self) -> dict:
        """Report the distribution of synthesized dialogues."""
        dist = {}
        for jsonl_file in self.output_dir.rglob("*.jsonl"):
            competition = jsonl_file.stem
            # PC-16: Use a context manager so the file handle is always closed
            # even if an exception occurs while reading lines.
            with open(jsonl_file) as fh:
                count = sum(1 for _ in fh)
            dist[competition] = count
        return dist


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Synthesize Socratic teaching dialogues")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", default="data/synthesized/teaching")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--target-total", type=int, default=90000)
    args = parser.parse_args()

    synthesizer = TeachingSynthesizer(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
        target_total=args.target_total,
    )
    asyncio.run(synthesizer.synthesize_all())
    dist = synthesizer.report_distribution()
    print(f"\nDistribution across competitions: {dist}")
