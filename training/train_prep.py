"""
train_prep.py — Data preparation before training

Merges all synthesized data, deduplicates, filters by quality,
and writes final train/val splits to data/train/.

Run before train.py:
    python training/train_prep.py --data-dir data --output-dir data/train
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from loguru import logger


def load_all_synthesized(synth_dir: Path) -> list[dict]:
    """Load all synthesized examples."""
    examples = []
    for jsonl_file in synth_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    logger.info(f"Loaded {len(examples):,} total examples")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove near-duplicate examples by problem statement hash."""
    seen = set()
    unique = []
    for ex in examples:
        conversations = ex.get("conversations", [])
        user_msg = next(
            (c.get("content", "") for c in conversations if c.get("role") == "user"),
            ""
        )
        key = hashlib.md5(user_msg[:200].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    logger.info(f"After dedup: {len(unique):,} unique examples")
    return unique


def filter_quality(examples: list[dict], min_score: float = 0.5) -> list[dict]:
    """Filter by quality score if available."""
    filtered = [
        ex for ex in examples
        if ex.get("metadata", {}).get("quality_score", 1.0) >= min_score
    ]
    logger.info(f"After quality filter: {len(filtered):,} examples")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    synth_dir = Path(args.data_dir) / "synthesized"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process
    examples = load_all_synthesized(synth_dir)
    examples = deduplicate(examples)
    examples = filter_quality(examples, args.min_quality)

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_fraction))
    train = examples[:split_idx]
    val = examples[split_idx:]

    # Write
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(output_dir / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    logger.success(
        f"Train: {len(train):,} | Val: {len(val):,} | "
        f"Written to {output_dir}"
    )


if __name__ == "__main__":
    main()
