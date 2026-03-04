"""
ProofCoach — Master Pipeline

Orchestrates the full data → train → evaluate pipeline.

Usage:
    python pipeline.py --collect          Download all competition data and AoPS solutions
    python pipeline.py --synthesize       Generate Socratic tutoring dialogues
    python pipeline.py --verify-lean      Batch-verify all claims with Lean 4
    python pipeline.py --train            Run all 3 training stages
    python pipeline.py --evaluate         Run CoachBench evaluation
    python pipeline.py --stats            Print dataset statistics
    python pipeline.py --tutor            Interactive tutoring session (local model)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/pipeline_{time}.log", rotation="100 MB")

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))
SYNTH_DIR = Path(os.getenv("SYNTH_DATA_DIR", "./data/synthesized"))
TRAIN_DIR = Path(os.getenv("TRAIN_DATA_DIR", "./data/train"))


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

async def run_collect(args) -> None:
    """Download all raw data: AMC/AIME/USAMO/IMO, AoPS solutions, Putnam."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.olympiads or args.all_data:
        logger.info("Downloading AMC/AIME/USAMO/IMO archives...")
        from discovery.olympiad_downloader import OlympiadDownloader
        downloader = OlympiadDownloader(output_dir=RAW_DIR / "olympiads")
        await downloader.download_all_competitions()

    if args.aops or args.all_data:
        logger.info("Crawling AoPS wiki and forum solutions...")
        from discovery.aops_crawler import AoPSCrawler
        async with AoPSCrawler(output_dir=RAW_DIR / "aops", workers=args.workers) as crawler:
            await crawler.crawl_all()

    if args.putnam or args.all_data:
        logger.info("Downloading Putnam archives...")
        from discovery.olympiad_downloader import PutnamDownloader
        downloader = PutnamDownloader(output_dir=RAW_DIR / "putnam")
        await downloader.download_all()

    print_stats_for_dir(RAW_DIR)


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------

async def run_synthesize(args) -> None:
    """Generate Socratic tutoring dialogues and misconception pairs."""
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    vllm_urls = args.vllm_urls or []
    backend = args.backend or "claude"

    if not vllm_urls and backend == "vllm":
        logger.error("--vllm-urls required when --backend vllm")
        sys.exit(1)

    if args.teaching or args.all_synth:
        logger.info("Synthesizing Socratic tutoring dialogues...")
        from synthesis.teaching_synthesizer import TeachingSynthesizer
        synthesizer = TeachingSynthesizer(
            raw_dir=RAW_DIR,
            output_dir=SYNTH_DIR / "teaching",
            backend=backend,
            vllm_urls=vllm_urls,
            workers=args.workers,
        )
        await synthesizer.synthesize_all()

    if args.misconceptions or args.all_synth:
        logger.info("Generating misconception diagnosis pairs...")
        from synthesis.misconception_generator import MisconceptionGenerator
        generator = MisconceptionGenerator(
            raw_dir=RAW_DIR,
            output_dir=SYNTH_DIR / "misconceptions",
            backend=backend,
            vllm_urls=vllm_urls,
            workers=args.workers,
        )
        await generator.generate_all()

    if args.dpo_pairs or args.all_synth:
        logger.info("Generating DPO preference pairs...")
        from synthesis.generate_dpo_pairs import DPOPairGenerator
        generator = DPOPairGenerator(
            synth_dir=SYNTH_DIR,
            output_dir=SYNTH_DIR / "dpo",
            backend=backend,
            vllm_urls=vllm_urls,
        )
        await generator.generate_all()

    # Deduplicate and merge into training set
    logger.info("Deduplicating and merging into training set...")
    await merge_and_dedup(SYNTH_DIR, TRAIN_DIR)
    print_stats_for_dir(TRAIN_DIR)


async def merge_and_dedup(synth_dir: Path, train_dir: Path) -> None:
    """Merge synthesized files and deduplicate via MinHash."""
    import hashlib

    train_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []
    seen_hashes: set[str] = set()

    for jsonl_file in synth_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)

                # Hash on the problem statement for dedup
                content_key = ""
                for turn in example.get("conversations", []):
                    if turn.get("role") == "user":
                        content_key = turn["content"][:200]
                        break
                h = hashlib.md5(content_key.encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                all_examples.append(example)

    logger.info(f"Total unique examples after dedup: {len(all_examples):,}")

    # PC-14: Shuffle before splitting so the val set is a random sample of
    # all topics/competitions rather than a contiguous slice of the last
    # competition processed (which would be biased by crawl order).
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Split 90/10 train/val
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    with open(train_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(train_dir / "val.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Train: {len(train_examples):,} | Val: {len(val_examples):,}")


# ---------------------------------------------------------------------------
# Lean 4 Verification
# ---------------------------------------------------------------------------

async def run_verify_lean(args) -> None:
    """Batch-verify all mathematical claims in the training set with Lean 4."""
    from core.lean4_interface import Lean4Interface

    interface = Lean4Interface(timeout=int(os.getenv("LEAN4_TIMEOUT", 10)))

    train_file = TRAIN_DIR / "train.jsonl"
    verified_file = TRAIN_DIR / "train_verified.jsonl"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        sys.exit(1)

    total = verified = failed = 0
    with open(train_file) as fin, open(verified_file, "w") as fout:
        for line in fin:
            example = json.loads(line.strip())
            claims = extract_lean4_claims(example)

            all_verified = True
            for claim in claims:
                result = interface.verify(claim)
                if not result.success:
                    all_verified = False
                    failed += 1
                else:
                    verified += 1
                total += 1

            example["lean4_verified"] = all_verified
            fout.write(json.dumps(example) + "\n")

    logger.info(f"Lean 4 verification: {verified}/{total} claims verified ({100*verified/max(total,1):.1f}%)")
    logger.info(f"Examples with all claims verified: written to {verified_file}")


def extract_lean4_claims(example: dict) -> list[str]:
    """Extract Lean 4 proposition strings from an example."""
    claims = []
    for turn in example.get("conversations", []):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            # Look for lean4_claim markers in the content
            import re
            for match in re.finditer(r"```lean\n(.*?)\n```", content, re.DOTALL):
                claims.append(match.group(1).strip())
    return claims


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(args) -> None:
    """Run all 3 training stages via subprocess (requires DeepSpeed)."""
    import subprocess

    model = args.base_model or "Qwen/Qwen2.5-7B-Instruct"
    num_gpus = args.num_gpus or 18

    if args.sft or args.all_train:
        logger.info("Stage 1: SFT...")
        cmd = [
            "deepspeed", f"--num_gpus={num_gpus}",
            "training/train.py",
            "--deepspeed", "training/configs/ds_config.json",
            "--model", model,
            "--data-dir", str(TRAIN_DIR),
            "--output-dir", "checkpoints/proofcoach-sft",
        ]
        subprocess.run(cmd, check=True)

    if args.rl or args.all_train:
        logger.info("Stage 2: GRPO RL with Lean 4 reward...")
        cmd = [
            "deepspeed", f"--num_gpus={num_gpus}",
            "training/train_rl.py",
            "--base-model", "checkpoints/proofcoach-sft/final",
            "--output-dir", "checkpoints/proofcoach-rl",
        ]
        subprocess.run(cmd, check=True)

    if args.dpo or args.all_train:
        logger.info("Stage 3: DPO on teaching quality...")
        cmd = [
            "deepspeed", f"--num_gpus={num_gpus}",
            "training/train_dpo.py",
            "--base-model", "checkpoints/proofcoach-rl/final",
            "--output-dir", "checkpoints/proofcoach-final",
        ]
        subprocess.run(cmd, check=True)

    logger.success("Training complete. Final model at checkpoints/proofcoach-final")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def run_evaluate(args) -> None:
    """Run CoachBench evaluation."""
    from evaluation.coachbench import CoachBench

    model_path = args.model or "checkpoints/proofcoach-final"
    bench = CoachBench(model_path=model_path)

    if args.solving or args.all_eval:
        results = bench.run_solving()
        logger.info(f"AMC accuracy: {results['amc_accuracy']:.1%}")
        logger.info(f"AIME accuracy: {results['aime_accuracy']:.1%}")

    if args.teaching or args.all_eval:
        results = bench.run_teaching()
        logger.info(f"Teaching effectiveness: {results['improvement']:.3f}")

    if args.lean or args.all_eval:
        results = bench.run_lean_verification()
        logger.info(f"Lean 4 verification rate: {results['verification_rate']:.1%}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(args) -> None:
    """Print dataset statistics."""
    for data_dir in [RAW_DIR, SYNTH_DIR, TRAIN_DIR]:
        if data_dir.exists():
            print_stats_for_dir(data_dir)


def print_stats_for_dir(data_dir: Path) -> None:
    """Print statistics for a data directory."""
    total_files = 0
    total_lines = 0
    total_bytes = 0

    for f in data_dir.rglob("*.jsonl"):
        total_files += 1
        total_bytes += f.stat().st_size
        with open(f) as fp:
            total_lines += sum(1 for _ in fp)

    logger.info(
        f"{data_dir.name}: {total_files} files, "
        f"{total_lines:,} examples, "
        f"{total_bytes / 1e9:.2f} GB"
    )


# ---------------------------------------------------------------------------
# Interactive Tutor
# ---------------------------------------------------------------------------

def run_tutor(args) -> None:
    """Launch an interactive tutoring session using the local model."""
    from agents.tutor_agent import TutorAgent

    model_path = args.model or "checkpoints/proofcoach-final"
    agent = TutorAgent(model_path=model_path)

    print("\nProofCoach Interactive Tutor")
    print("=" * 50)
    print("Enter a math problem or type 'quit' to exit.\n")

    session_id = "interactive_001"
    while True:
        problem = input("Problem: ").strip()
        if problem.lower() in ("quit", "exit", "q"):
            break
        if not problem:
            continue

        student_work = input("Your work so far (press Enter to skip): ").strip()

        response = agent.tutor(
            problem=problem,
            student_work=student_work or None,
            session_id=session_id,
        )

        print(f"\nProofCoach: {response['question']}")
        if response.get("hint_level"):
            print(f"[Hint level: {response['hint_level']}/5]")
        if response.get("verified_steps"):
            print(f"[Lean 4 verified steps: {len(response['verified_steps'])}]")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ProofCoach Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Collect
    collect_parser = subparsers.add_parser("collect", help="Download raw data")
    collect_parser.add_argument("--all-data", action="store_true")
    collect_parser.add_argument("--olympiads", action="store_true")
    collect_parser.add_argument("--aops", action="store_true")
    collect_parser.add_argument("--putnam", action="store_true")
    collect_parser.add_argument("--workers", type=int, default=20)

    # Synthesize
    synth_parser = subparsers.add_parser("synthesize", help="Generate training data")
    synth_parser.add_argument("--all-synth", action="store_true")
    synth_parser.add_argument("--teaching", action="store_true")
    synth_parser.add_argument("--misconceptions", action="store_true")
    synth_parser.add_argument("--dpo-pairs", action="store_true")
    synth_parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    synth_parser.add_argument("--vllm-urls", nargs="+")
    synth_parser.add_argument("--workers", type=int, default=20)

    # Verify
    verify_parser = subparsers.add_parser("verify-lean", help="Batch Lean 4 verification")

    # Train
    train_parser = subparsers.add_parser("train", help="Run training stages")
    train_parser.add_argument("--all-train", action="store_true")
    train_parser.add_argument("--sft", action="store_true")
    train_parser.add_argument("--rl", action="store_true")
    train_parser.add_argument("--dpo", action="store_true")
    train_parser.add_argument("--base-model", type=str)
    train_parser.add_argument("--num-gpus", type=int, default=18)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run CoachBench")
    eval_parser.add_argument("--all-eval", action="store_true")
    eval_parser.add_argument("--solving", action="store_true")
    eval_parser.add_argument("--teaching", action="store_true")
    eval_parser.add_argument("--lean", action="store_true")
    eval_parser.add_argument("--model", type=str)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Print dataset stats")

    # Tutor (interactive)
    tutor_parser = subparsers.add_parser("tutor", help="Interactive tutoring session")
    tutor_parser.add_argument("--model", type=str)

    # Legacy flat flags (for backward compat with README quick-start)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--synthesize", action="store_true")
    parser.add_argument("--verify-lean", dest="verify_lean_flat", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--tutor", action="store_true")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--num-gpus", type=int, default=18)
    parser.add_argument("--model", type=str)
    parser.add_argument("--all", action="store_true")
    # Flat-flag sub-options (used when subcommand is not specified)
    parser.add_argument("--olympiads", action="store_true")
    parser.add_argument("--aops", action="store_true")
    parser.add_argument("--putnam", action="store_true")
    parser.add_argument("--teaching", action="store_true")
    parser.add_argument("--misconceptions", action="store_true")
    parser.add_argument("--dpo-pairs", dest="dpo_pairs", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    parser.add_argument("--base-model", dest="base_model", type=str)
    parser.add_argument("--solving", action="store_true")
    parser.add_argument("--lean", action="store_true")

    args = parser.parse_args()

    # Handle flat flags
    if args.collect or args.all:
        args.all_data = getattr(args, 'all_data', False) or args.all
        asyncio.run(run_collect(args))

    if args.synthesize or args.all:
        args.all_synth = getattr(args, 'all_synth', False) or args.all
        asyncio.run(run_synthesize(args))

    if args.verify_lean_flat or args.all:
        asyncio.run(run_verify_lean(args))

    if args.train or args.all:
        args.all_train = getattr(args, 'all_train', False) or args.all
        run_train(args)

    if args.evaluate or args.all:
        args.all_eval = getattr(args, 'all_eval', False) or args.all
        run_evaluate(args)

    if args.stats:
        print_stats(args)

    if args.tutor:
        run_tutor(args)

    if not any([args.collect, args.synthesize, args.verify_lean_flat,
                args.train, args.evaluate, args.stats, args.tutor, args.all,
                args.command]):
        parser.print_help()


if __name__ == "__main__":
    main()
