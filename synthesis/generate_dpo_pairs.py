"""
DPO Pair Generator — Teaching quality preference pairs

Generates (chosen, rejected) pairs where:
  chosen:   Socratic guidance that leads student to insight
  rejected: Direct answer / mechanical solution / vague response

Preference signals:
  - AoPS community upvotes (high-voted tutoring post = chosen)
  - Generating rejected responses via LLM
  - Rule-based quality scoring
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for generate_dpo_pairs. "
        "Install with: pip install aiofiles"
    ) from exc
from loguru import logger

from synthesis.synthesize_bulk import SynthesisPipeline
from synthesis.prompts import GENERATE_REJECTED_SYSTEM, GENERATE_REJECTED_USER


class DPOPairGenerator:
    """Generates DPO preference pairs for teaching quality."""

    def __init__(
        self,
        synth_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        target_pairs: int = 20000,
    ) -> None:
        self.synth_dir = Path(synth_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.target_pairs = target_pairs

        # PC-17: Pass the dedicated output_dir (not synth_dir) as the helper's
        # output directory. Using synth_dir for both raw_dir and output_dir
        # would cause the helper to write into the input source directory,
        # mixing generated DPO pairs with the source training data.
        self._helper = SynthesisPipeline(
            raw_dir=synth_dir,
            output_dir=self.output_dir,
            backend=backend,
            vllm_urls=vllm_urls,
            workers=20,
        )
        self._stats = {"generated": 0, "failed": 0}

    async def generate_all(self) -> None:
        """Generate DPO pairs from synthesized teaching dialogues."""
        logger.info(f"Generating DPO pairs (target: {self.target_pairs:,})...")

        # Load synthesized teaching dialogues (these become "chosen")
        chosen_examples = self._load_synthesized_dialogues()
        logger.info(f"Loaded {len(chosen_examples):,} synthesized dialogues as chosen examples")

        output_file = self.output_dir / "dpo_pairs.jsonl"

        # Generate rejected counterparts
        tasks = [self._generate_pair(ex) for ex in chosen_examples[:self.target_pairs]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, dict):
                    await f.write(json.dumps(result) + "\n")
                    self._stats["generated"] += 1

        logger.info(f"DPO pairs complete: {self._stats['generated']:,} pairs")

    def _load_synthesized_dialogues(self) -> list[dict]:
        """Load the synthesized teaching dialogues."""
        dialogues = []
        for jsonl_file in self.synth_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ex = json.loads(line)
                        # Only use high-quality examples as chosen
                        if ex.get("metadata", {}).get("quality_score", 0) >= 0.7:
                            dialogues.append(ex)
        return dialogues

    async def _generate_pair(self, chosen_example: dict) -> Optional[dict]:
        """Generate a (chosen, rejected) pair."""
        conversations = chosen_example.get("conversations", [])
        if len(conversations) < 2:
            return None

        # Find the user message
        user_msg = next(
            (c["content"] for c in conversations if c.get("role") == "user"), ""
        )
        chosen_response = next(
            (c["content"] for c in conversations if c.get("role") == "assistant"), ""
        )

        if not user_msg or not chosen_response:
            return None

        # Generate a rejected (low-quality) response
        rejected_response = await self._generate_rejected(user_msg)
        if not rejected_response:
            # Fallback: use a simple "here's the answer" template
            metadata = chosen_example.get("metadata", {})
            rejected_response = (
                f"The answer to this problem is {metadata.get('answer', 'the solution')}. "
                f"You can solve it using {metadata.get('approach_name', 'the standard method')}."
            )

        return {
            "prompt": [
                c for c in conversations if c.get("role") in ("system", "user")
            ],
            "chosen": [{"role": "assistant", "content": chosen_response}],
            "rejected": [{"role": "assistant", "content": rejected_response}],
            "metadata": {
                "problem_id": chosen_example.get("metadata", {}).get("problem_id", ""),
                "chosen_type": "socratic_dialogue",
                "rejected_type": "direct_answer",
            },
        }

    async def _generate_rejected(self, user_message: str) -> Optional[str]:
        """Generate a low-quality rejected response."""
        prompt = GENERATE_REJECTED_USER.format(
            problem_statement=user_message,
            student_attempt="",
        )
        return await self._helper._call_llm(
            GENERATE_REJECTED_SYSTEM, prompt, max_tokens=300
        )
