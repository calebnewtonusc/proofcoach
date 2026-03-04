"""
lean4_verifier.py — Validate Lean 4 proof syntax and quality.

Two levels of verification:
  1. Syntax check: balanced brackets, valid keywords, basic structure (no Lean binary needed)
  2. Full verification: runs `lean --check` on a temp file (requires Lean 4 installed)

Also scores proof quality:
  - Elegant (short + few tactics) vs verbose
  - Uses modern Lean 4 tactics (omega, simp, ring) vs decide-everything
  - Proof compiles without sorry

Usage:
    python validation/lean4_verifier.py --input data/raw/lean4/mathlib4.jsonl
    python validation/lean4_verifier.py --input data/synthesized/lean4.jsonl --full
"""

import asyncio
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from loguru import logger


OUTPUT_DIR = Path(__file__).parents[1] / "data" / "verified"

# ── Lean 4 known keywords ────────────────────────────────────────────────────

LEAN4_TACTICS = {
    # Core tactics
    "intro", "intros", "apply", "exact", "refine", "rfl", "rfl?",
    "simp", "simp?", "simp_all", "simp_all?",
    "ring", "ring_nf", "ring?",
    "omega", "linarith", "nlinarith",
    "norm_num", "norm_num?", "norm_cast", "push_cast",
    "decide", "native_decide",
    "trivial", "tauto", "aesop",
    "contradiction", "absurd", "exfalso",
    "constructor", "left", "right",
    "cases", "rcases", "obtain",
    "induction", "induction_on",
    "have", "let", "suffices", "show",
    "use", "exists", "use_or",
    "rw", "rw?", "rwa",
    "unfold", "delta", "change",
    "conv", "congr",
    "split", "if_pos", "if_neg",
    "push_neg", "contrapose",
    "ext", "funext", "iff_intro",
    "calc", "trans", "symm",
    "assumption", "assumption?",
    "done", "sorry",
    "first", "try", "repeat", "iterate",
    "all_goals", "any_goals", "on_goal",
    "focus", "rotate_left", "rotate_right",
    "field_simp", "positivity",
    "gcongr", "mono",
}

# Modern preferred tactics (higher quality signal)
MODERN_TACTICS = {"omega", "ring", "simp", "linarith", "norm_num", "aesop", "gcongr", "positivity"}

# Anti-patterns
SORRY_PATTERN = re.compile(r'\bsorry\b')
ADMITTED_PATTERN = re.compile(r'\badmit\b')

# Lean 4 structure patterns
LEAN4_PROOF_KEYWORDS = re.compile(
    r'\b(by|theorem|lemma|proposition|def|noncomputable|protected|private)\b'
)

LEAN4_BLOCK_START = re.compile(r'\bby\s*$|\bby\s+\w')
LEAN4_TERM_PROOF = re.compile(r':=\s*(?!by)')


@dataclass
class VerificationResult:
    """Result of verifying a single Lean 4 proof."""
    theorem_id: str
    theorem_name: str
    statement: str
    proof: str

    # Verification levels
    syntax_valid: bool
    syntax_errors: list[str]
    binary_verified: Optional[bool]  # None if Lean binary not available
    binary_error: Optional[str]

    # Quality metrics
    quality_score: float       # 0.0 – 1.0
    has_sorry: bool
    proof_type: str            # tactic | term | decide | sorry | unknown
    tactics_used: list[str]
    proof_length_lines: int
    is_elegant: bool           # short proof using modern tactics

    # Metadata
    verification_time_ms: float
    lean_version: Optional[str]


def detect_lean_binary() -> Optional[str]:
    """Detect Lean 4 binary on PATH and return version string."""
    for binary in ["lean", "lean4"]:
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


LEAN_VERSION: Optional[str] = None
LEAN_BINARY_AVAILABLE: bool = False


def _init_lean_binary() -> None:
    global LEAN_VERSION, LEAN_BINARY_AVAILABLE
    LEAN_VERSION = detect_lean_binary()
    LEAN_BINARY_AVAILABLE = LEAN_VERSION is not None
    if LEAN_BINARY_AVAILABLE:
        logger.info(f"Lean binary found: {LEAN_VERSION}")
    else:
        logger.info("Lean binary not found — syntax-only validation mode")


def check_syntax(lean_code: str) -> tuple[bool, list[str]]:
    """
    Structural syntax check for Lean 4 code without a binary.

    Checks:
      - Balanced parentheses, brackets, braces, angle brackets (in type context)
      - No unclosed string literals
      - Presence of Lean 4 structural keywords
      - Not purely empty
    """
    errors: list[str] = []

    if not lean_code or not lean_code.strip():
        return False, ["Empty proof"]

    # Check for balanced delimiters
    stack: list[tuple[str, int]] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    close_to_open = {v: k for k, v in pairs.items()}

    in_string = False
    in_comment = False
    lines = lean_code.splitlines()

    for line_num, line in enumerate(lines, 1):
        i = 0
        while i < len(line):
            ch = line[i]

            # Line comment
            if not in_string and i + 1 < len(line) and line[i:i+2] == "--":
                break  # rest of line is comment

            # Block comment start: /-
            if not in_string and i + 1 < len(line) and line[i:i+2] == "/-":
                in_comment = True
                i += 2
                continue

            # Block comment end: -/
            if in_comment and i + 1 < len(line) and line[i:i+2] == "-/":
                in_comment = False
                i += 2
                continue

            if in_comment:
                i += 1
                continue

            # String literal
            if ch == '"' and not in_string:
                in_string = True
                i += 1
                continue
            if ch == '"' and in_string:
                in_string = False
                i += 1
                continue

            if in_string:
                i += 1
                continue

            # Bracket tracking
            if ch in pairs:
                stack.append((ch, line_num))
            elif ch in close_to_open:
                expected_open = close_to_open[ch]
                if not stack:
                    errors.append(f"Line {line_num}: Unmatched closing '{ch}'")
                elif stack[-1][0] != expected_open:
                    errors.append(
                        f"Line {line_num}: Expected '{pairs[stack[-1][0]]}' but got '{ch}'"
                    )
                    stack.pop()
                else:
                    stack.pop()

            i += 1

    # Unclosed delimiters
    for open_ch, open_line in stack:
        errors.append(f"Line {open_line}: Unclosed '{open_ch}'")

    if in_string:
        errors.append("Unclosed string literal")

    # Check for minimum Lean 4 structure
    has_keywords = bool(LEAN4_PROOF_KEYWORDS.search(lean_code))
    has_definition = bool(re.search(r'\b(?:theorem|lemma|def|proposition)\s+\w', lean_code))

    if not has_keywords and not has_definition:
        errors.append("No Lean 4 structural keywords found (theorem/lemma/def/by)")

    # Check for sorry (not an error, but flagged)
    if SORRY_PATTERN.search(lean_code):
        errors.append("WARNING: proof contains 'sorry' (incomplete proof)")

    syntax_valid = len([e for e in errors if not e.startswith("WARNING")]) == 0
    return syntax_valid, errors


def verify_with_lean_binary(lean_code: str, timeout: int = 30) -> tuple[bool, Optional[str]]:
    """
    Verify a Lean 4 proof using the lean binary.

    Writes code to a temp file and runs `lean --check`.
    Returns (success, error_message).
    """
    if not LEAN_BINARY_AVAILABLE:
        return False, "Lean binary not available"

    # Wrap in a minimal Lean 4 file structure if not already a full file
    if "import" not in lean_code and "theorem" in lean_code:
        full_code = f"import Mathlib\nimport Mathlib.Tactic\n\n{lean_code}"
    else:
        full_code = lean_code

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            prefix="proofcoach_verify_",
            delete=False,
        ) as f:
            f.write(full_code)
            temp_path = f.name

        result = subprocess.run(
            ["lean", "--check", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        Path(temp_path).unlink(missing_ok=True)

        if result.returncode == 0:
            return True, None
        else:
            error = (result.stderr or result.stdout).strip()
            # Trim very long error messages
            if len(error) > 500:
                error = error[:500] + "...(truncated)"
            return False, error

    except subprocess.TimeoutExpired:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"Lean verification timed out after {timeout}s"
    except Exception as e:
        return False, f"Lean binary error: {e}"


def extract_tactics(proof: str) -> list[str]:
    """Extract tactic names used in a Lean 4 proof."""
    found = []
    for tactic in LEAN4_TACTICS:
        pattern = rf'\b{re.escape(tactic)}\b'
        if re.search(pattern, proof):
            found.append(tactic)
    return sorted(found)


def detect_proof_type(proof: str) -> str:
    """Classify proof type: tactic | term | decide | sorry | unknown."""
    if SORRY_PATTERN.search(proof) or ADMITTED_PATTERN.search(proof):
        return "sorry"
    if "native_decide" in proof:
        return "decide"
    if re.search(r'\bdecide\b(?!\s*=)', proof) and "by" not in proof.split("decide")[0][-20:]:
        return "decide"
    if "by" in proof:
        return "tactic"
    if ":=" in proof:
        return "term"
    return "unknown"


def score_proof_quality(
    proof: str,
    is_syntax_valid: bool,
    is_binary_verified: Optional[bool],
    tactics: list[str],
) -> tuple[float, bool]:
    """
    Score proof quality from 0.0 to 1.0.

    Scoring:
      - Binary verified: +0.50
      - Syntax valid:    +0.20
      - No sorry:        +0.10
      - Modern tactics:  +0.10 (omega, ring, simp, linarith)
      - Elegant (concise): +0.10

    Returns (score, is_elegant).
    """
    score = 0.0

    if is_binary_verified is True:
        score += 0.50
    elif is_syntax_valid:
        score += 0.20

    has_sorry = bool(SORRY_PATTERN.search(proof))
    if not has_sorry:
        score += 0.10

    modern_used = set(tactics) & MODERN_TACTICS
    if modern_used:
        score += 0.10

    # Elegance: short proof with modern tactics
    proof_lines = [l for l in proof.splitlines() if l.strip()]
    line_count = len(proof_lines)
    is_elegant = line_count <= 10 and bool(modern_used) and not has_sorry

    if is_elegant:
        score += 0.10

    return round(min(1.0, score), 3), is_elegant


def verify_proof(
    theorem_id: str,
    theorem_name: str,
    statement: str,
    proof: str,
    use_binary: bool = False,
) -> VerificationResult:
    """
    Verify a single Lean 4 proof.

    Args:
        theorem_id: Unique identifier
        theorem_name: Name of the theorem
        statement: Full theorem statement
        proof: Lean 4 proof code
        use_binary: Whether to attempt Lean binary verification
    """
    start = time.monotonic()

    # Full code = statement + proof
    full_code = f"{statement}\n{proof}" if proof and statement not in proof else proof

    # Level 1: syntax check
    syntax_valid, syntax_errors = check_syntax(full_code)

    # Level 2: binary verification (optional)
    binary_verified: Optional[bool] = None
    binary_error: Optional[str] = None

    if use_binary and LEAN_BINARY_AVAILABLE:
        binary_verified, binary_error = verify_with_lean_binary(full_code)

    # Extract metadata
    tactics = extract_tactics(proof)
    proof_type = detect_proof_type(proof)
    has_sorry = bool(SORRY_PATTERN.search(proof))
    proof_lines = [l for l in proof.splitlines() if l.strip()]

    quality_score, is_elegant = score_proof_quality(
        proof, syntax_valid, binary_verified, tactics
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    return VerificationResult(
        theorem_id=theorem_id,
        theorem_name=theorem_name,
        statement=statement[:1000],
        proof=proof[:2000],
        syntax_valid=syntax_valid,
        syntax_errors=[e for e in syntax_errors if not e.startswith("WARNING")],
        binary_verified=binary_verified,
        binary_error=binary_error,
        quality_score=quality_score,
        has_sorry=has_sorry,
        proof_type=proof_type,
        tactics_used=tactics,
        proof_length_lines=len(proof_lines),
        is_elegant=is_elegant,
        verification_time_ms=round(elapsed_ms, 2),
        lean_version=LEAN_VERSION,
    )


class Lean4VerifierBatch:
    """
    Batch verifier for Lean 4 theorem-proof pairs.

    Reads from synthesized data (lean4_synthesizer.py output) or
    from raw crawler data (lean4_mathlib.py output).
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        use_binary: bool = False,
        min_quality_score: float = 0.3,
        workers: int = 8,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = output_path or (
            OUTPUT_DIR / f"{self.input_path.stem}_verified.jsonl"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_binary = use_binary
        self.min_quality_score = min_quality_score
        self.workers = workers
        self._stats = {
            "total": 0,
            "syntax_valid": 0,
            "binary_verified": 0,
            "has_sorry": 0,
            "elegant": 0,
            "passed_filter": 0,
        }

    def _load_records(self) -> list[dict]:
        """Load JSONL records. Supports both synthesized and raw formats."""
        records = []
        with open(self.input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _extract_theorem_proof(self, record: dict) -> tuple[str, str, str, str]:
        """
        Extract (theorem_id, theorem_name, statement, proof) from various record formats.

        Supports:
          - lean4_mathlib.py format: theorem_id, theorem_name, statement, proof
          - lean4_synthesizer.py format: conversations list
        """
        # Direct format (from lean4_mathlib.py crawler)
        if "theorem_name" in record and "statement" in record:
            return (
                record.get("theorem_id", record["theorem_name"]),
                record["theorem_name"],
                record.get("statement", ""),
                record.get("proof", ""),
            )

        # Conversation format (from lean4_synthesizer.py)
        if "conversations" in record:
            convs = record["conversations"]
            theorem_id = record.get("metadata", {}).get("theorem_id", "unknown")
            theorem_name = record.get("metadata", {}).get("theorem_name", "unknown")

            # User turn contains the problem/sketch, assistant contains the proof
            statement = ""
            proof = ""
            for turn in convs:
                role = turn.get("role") or turn.get("from", "")
                content = turn.get("content") or turn.get("value", "")
                if role in ("user", "human"):
                    statement = content
                elif role in ("assistant", "gpt"):
                    proof = content

            # Extract lean4 code from assistant response
            lean_match = re.search(r'```(?:lean4?|lean)\n(.*?)```', proof, re.DOTALL)
            if lean_match:
                proof = lean_match.group(1).strip()

            return theorem_id, theorem_name, statement, proof

        # Fallback: use whatever fields are present
        return (
            record.get("id", "unknown"),
            record.get("name", "unknown"),
            record.get("statement", record.get("query", "")),
            record.get("proof", record.get("response", "")),
        )

    async def _verify_batch_async(self, records: list[dict]) -> list[VerificationResult]:
        """Verify records in bounded-concurrency async batches."""
        semaphore = asyncio.Semaphore(self.workers)
        # PC-20: asyncio.get_event_loop() is deprecated in Python 3.10+ when
        # called from within a running coroutine. Use asyncio.get_running_loop()
        # instead, which is always safe inside an async context.
        loop = asyncio.get_running_loop()

        async def verify_one(record: dict) -> Optional[VerificationResult]:
            async with semaphore:
                tid, tname, stmt, proof = self._extract_theorem_proof(record)
                if not proof or not proof.strip():
                    return None
                # Run CPU-bound verification in thread pool
                result = await loop.run_in_executor(
                    None,
                    verify_proof,
                    tid, tname, stmt, proof, self.use_binary,
                )
                return result

        tasks = [verify_one(r) for r in records]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in raw_results if isinstance(r, VerificationResult)]

    def verify_all(self) -> dict:
        """Run verification on all records and write filtered output."""
        _init_lean_binary()

        logger.info(f"Loading records from {self.input_path}")
        records = self._load_records()
        logger.info(f"Loaded {len(records):,} records")

        results = asyncio.run(self._verify_batch_async(records))

        self._stats["total"] = len(results)

        with open(self.output_path, "w") as out_f:
            for result in results:
                if result.syntax_valid:
                    self._stats["syntax_valid"] += 1
                if result.binary_verified is True:
                    self._stats["binary_verified"] += 1
                if result.has_sorry:
                    self._stats["has_sorry"] += 1
                if result.is_elegant:
                    self._stats["elegant"] += 1

                if result.quality_score >= self.min_quality_score:
                    self._stats["passed_filter"] += 1
                    out_f.write(json.dumps(asdict(result)) + "\n")

        logger.info(
            f"Verification complete:\n"
            f"  Total:          {self._stats['total']:>6,}\n"
            f"  Syntax valid:   {self._stats['syntax_valid']:>6,}\n"
            f"  Binary verified:{self._stats['binary_verified']:>6,}\n"
            f"  Has sorry:      {self._stats['has_sorry']:>6,}\n"
            f"  Elegant:        {self._stats['elegant']:>6,}\n"
            f"  Passed filter:  {self._stats['passed_filter']:>6,} "
            f"(score >= {self.min_quality_score})"
        )
        logger.success(f"Output: {self.output_path}")
        return self._stats


def verify_single(lean_code: str, use_binary: bool = False) -> dict:
    """
    Convenience function for single proof verification.

    Returns verification result as a dict.
    """
    _init_lean_binary()
    result = verify_proof(
        theorem_id="inline",
        theorem_name="inline",
        statement="",
        proof=lean_code,
        use_binary=use_binary,
    )
    return asdict(result)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Verify Lean 4 proofs from JSONL")
    parser.add_argument(
        "--input", required=True,
        help="Input JSONL file (lean4_mathlib.py or lean4_synthesizer.py output)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL file (default: data/verified/<input_stem>_verified.jsonl)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use Lean binary for full verification (requires Lean 4 installed)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.3,
        help="Minimum quality score to include in output (default: 0.3)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Async workers for parallel verification (default: 8)",
    )
    parser.add_argument(
        "--single", default=None,
        help="Verify a single inline Lean 4 proof string",
    )
    args = parser.parse_args()

    if args.single:
        result = verify_single(args.single, use_binary=args.full)
        print(json.dumps(result, indent=2))
    else:
        verifier = Lean4VerifierBatch(
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            use_binary=args.full,
            min_quality_score=args.min_score,
            workers=args.workers,
        )
        stats = verifier.verify_all()
        print(f"\nVerification summary: {stats}")
