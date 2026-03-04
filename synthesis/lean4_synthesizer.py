"""
lean4_synthesizer.py — Synthesize Lean 4 formal proofs from informal proof sketches.

Given (informal_proof_sketch), uses vLLM/Claude to generate Lean 4 proof code.
Validates syntax at minimum; invokes Lean 4 binary when available.

Workflow:
  1. Load (informal_proof, theorem_statement) pairs from data/raw/lean4/
  2. For each pair: prompt LLM to write a Lean 4 proof
  3. Run lean --check on the output (if Lean 4 is installed)
  4. Score proof quality: syntax_valid, lean_verified, proof_length (elegance proxy)
  5. Save (informal_proof, formal_lean4_proof, quality) records

Output: data/synthesized/lean4_proofs.jsonl

Usage:
    python synthesis/lean4_synthesizer.py --backend vllm \
        --vllm-urls http://localhost:8001 http://localhost:8002
    python synthesis/lean4_synthesizer.py --backend claude --workers 10
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# PC-15: Import aiofiles directly; the HAS_AIOFILES flag was dead code because
# the except block immediately re-raised the ImportError, making the False
# branch unreachable. Fail loudly on import if the package is missing.
try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for lean4_synthesizer. "
        "Install with: pip install aiofiles"
    ) from exc
import aiohttp
import anthropic
from loguru import logger


OUTPUT_DIR = Path(__file__).parents[1] / "data" / "synthesized"
INPUT_DIR  = Path(__file__).parents[1] / "data" / "raw" / "lean4"

LEAN4_SYSTEM_PROMPT = """You are an expert Lean 4 (Lean4) proof writer with deep knowledge of Mathlib4.

Given an informal mathematical proof sketch and the theorem statement in Lean 4 syntax, write a complete, compilable Lean 4 proof.

Rules:
1. Use standard Mathlib4 tactics: simp, ring, linarith, omega, norm_num, exact, apply, intro, constructor, ext
2. Import only what is needed: import Mathlib.Tactic or specific imports
3. The proof must type-check — no sorry, no admit
4. Prefer shorter elegant proofs over verbose ones
5. When the informal proof uses "by induction", use the `induction` tactic
6. When the informal proof uses "by contradiction", use `by_contra`
7. Always close tactic blocks properly

Output format:
```lean4
import Mathlib.Tactic
-- theorem statement (as given)
-- proof
```"""

LEAN4_USER_TEMPLATE = """Informal proof sketch:
{informal_proof}

Lean 4 theorem statement:
```lean4
{theorem_statement}
```

Write the complete compilable Lean 4 proof for this theorem."""


@dataclass
class Lean4ProofResult:
    """Result of synthesizing a Lean 4 proof."""
    theorem_id: str
    theorem_name: str
    theorem_statement: str
    informal_proof: str
    lean4_proof: str
    syntax_valid: bool
    lean_verified: bool
    proof_length: int      # number of lines (shorter = more elegant)
    quality_score: float
    tactics_used: list[str]
    error_message: Optional[str] = None


def check_lean4_available() -> bool:
    """Check if the Lean 4 binary is available."""
    return shutil.which("lean") is not None or shutil.which("lake") is not None


def validate_lean4_syntax(lean_code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Lean 4 proof syntax.

    Returns (is_valid, error_message).
    If Lean 4 binary is available: actually verifies the proof.
    Otherwise: performs lightweight syntax checking heuristics.
    """
    # Lightweight syntax check (always run)
    # Check for common indicators of invalid code
    if "sorry" in lean_code.lower() or "admit" in lean_code.lower():
        return False, "Proof uses sorry/admit placeholder"

    if not any(keyword in lean_code for keyword in ["theorem", "lemma", "proposition", "def", "example"]):
        return False, "No theorem declaration found"

    # Check balanced angle brackets (common Lean 4 issue)
    angle_open = lean_code.count("<")
    angle_close = lean_code.count(">")
    paren_open = lean_code.count("(")
    paren_close = lean_code.count(")")
    bracket_open = lean_code.count("[")
    bracket_close = lean_code.count("]")
    curly_open = lean_code.count("{")
    curly_close = lean_code.count("}")

    if paren_open != paren_close:
        return False, f"Unbalanced parentheses: {paren_open} open, {paren_close} close"
    if bracket_open != bracket_close:
        return False, f"Unbalanced brackets"
    if curly_open != curly_close:
        return False, f"Unbalanced curly braces"

    # Try running with Lean 4 if available
    if check_lean4_available():
        return _verify_with_lean_binary(lean_code)

    return True, None


def _verify_with_lean_binary(lean_code: str) -> tuple[bool, Optional[str]]:
    """Verify proof using the Lean 4 binary."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(lean_code)
        tmp_path = f.name

    try:
        lean_bin = shutil.which("lean") or "lean"
        result = subprocess.run(
            [lean_bin, "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, None
        else:
            error = result.stderr.strip()[:500]
            return False, error
    except subprocess.TimeoutExpired:
        return False, "Lean check timed out"
    except Exception as e:
        return False, f"Lean binary error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_lean4_proof(response: str) -> Optional[str]:
    """Extract Lean 4 proof from LLM response."""
    # Look for lean4 or lean code blocks
    patterns = [
        r"```lean4?\s*([\s\S]+?)\s*```",
        r"```\s*(import[\s\S]+?)\s*```",
    ]
    for pattern in patterns:
        m = re.search(pattern, response, re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if len(code) > 30:
                return code

    # Fallback: look for lines starting with import or theorem
    lines = response.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("theorem ") or stripped.startswith("lemma "):
            in_code = True
        if in_code:
            code_lines.append(line)
            # Stop at obvious end markers
            if stripped == "" and len(code_lines) > 5:
                break

    if code_lines and len("\n".join(code_lines)) > 30:
        return "\n".join(code_lines)

    return None


def score_proof(lean_code: str, is_verified: bool, is_syntax_valid: bool) -> float:
    """Score proof quality 0.0-1.0."""
    score = 0.0

    if not is_syntax_valid:
        return 0.0

    if is_verified:
        score += 0.5
    else:
        score += 0.2  # Partial credit for syntactically valid

    # Elegance: shorter proofs are better
    lines = [l for l in lean_code.splitlines() if l.strip() and not l.strip().startswith("--")]
    n_lines = len(lines)
    if n_lines <= 5:
        score += 0.3
    elif n_lines <= 10:
        score += 0.2
    elif n_lines <= 20:
        score += 0.1
    else:
        score += 0.05

    # Uses good tactics (proof quality indicator)
    good_tactics = ["ring", "omega", "simp", "norm_num", "linarith", "decide"]
    tactics_used = [t for t in good_tactics if t in lean_code]
    if tactics_used:
        score += min(0.2, 0.05 * len(tactics_used))

    # Doesn't have sorry
    if "sorry" not in lean_code and "admit" not in lean_code:
        score += 0.1

    return min(1.0, score)


def extract_tactics(lean_code: str) -> list[str]:
    """Extract tactic names used in the proof."""
    all_tactics = [
        "simp", "ring", "linarith", "omega", "norm_num", "decide",
        "exact", "apply", "intro", "constructor", "ext", "rfl",
        "cases", "induction", "rcases", "obtain", "have", "show",
        "by_contra", "by_cases", "push_neg", "use", "refine",
        "field_simp", "ring_nf", "positivity", "bound",
    ]
    code_lower = lean_code.lower()
    return [t for t in all_tactics if re.search(r"\b" + re.escape(t) + r"\b", code_lower)]


class Lean4Synthesizer:
    """Synthesizes Lean 4 formal proofs from informal proof sketches."""

    def __init__(
        self,
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 15,
        min_quality: float = 0.3,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self.min_quality = min_quality
        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_idx = 0
        self._lean4_available = check_lean4_available()
        self._stats = {
            "attempted": 0,
            "syntax_valid": 0,
            "lean_verified": 0,
            "failed": 0,
            "quality_filtered": 0,
        }

        if backend == "claude":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )

        if self._lean4_available:
            logger.info("Lean 4 binary found — will verify proofs")
        else:
            logger.info("Lean 4 binary NOT found — using syntax-only validation")

    async def _call_llm(
        self,
        user: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """Call LLM backend."""
        if self.backend == "claude":
            try:
                resp = await self._anthropic.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=2000,
                    system=LEAN4_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except Exception as e:
                logger.debug(f"Claude error: {e}")
                return None
        else:
            if not self.vllm_urls or not session:
                return None
            url = self.vllm_urls[self._vllm_idx % len(self.vllm_urls)]
            self._vllm_idx += 1
            try:
                async with session.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen2.5-72B-Instruct",
                        "messages": [
                            {"role": "system", "content": LEAN4_SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.3,
                    },
                    headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'synthesis')}"},
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    return None
            except Exception as e:
                logger.debug(f"vLLM error: {e}")
                return None

    async def _synthesize_one(
        self,
        theorem: dict,
        output_file: Path,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[Lean4ProofResult]:
        """Synthesize a Lean 4 proof for a single theorem."""
        async with self._semaphore:
            self._stats["attempted"] += 1

            theorem_id = theorem.get("theorem_id", "unknown")
            theorem_name = theorem.get("theorem_name", "unknown")
            statement = theorem.get("statement", "")
            existing_proof = theorem.get("proof", "")

            if not statement or len(statement) < 20:
                self._stats["failed"] += 1
                return None

            # Use existing proof as informal sketch
            informal = existing_proof if existing_proof else f"Prove the theorem: {statement}"

            user_prompt = LEAN4_USER_TEMPLATE.format(
                informal_proof=informal[:2000],
                theorem_statement=statement[:1500],
            )

            response = await self._call_llm(user_prompt, session)
            if not response:
                self._stats["failed"] += 1
                return None

            lean4_proof = extract_lean4_proof(response)
            if not lean4_proof:
                self._stats["failed"] += 1
                return None

            # Validate
            syntax_valid, error_msg = validate_lean4_syntax(lean4_proof)
            lean_verified = False

            if syntax_valid:
                self._stats["syntax_valid"] += 1
                if self._lean4_available:
                    lean_verified, error_msg = _verify_with_lean_binary(lean4_proof)
                    if lean_verified:
                        self._stats["lean_verified"] += 1

            quality = score_proof(lean4_proof, lean_verified, syntax_valid)

            if quality < self.min_quality:
                self._stats["quality_filtered"] += 1
                return None

            tactics = extract_tactics(lean4_proof)
            n_lines = len([l for l in lean4_proof.splitlines() if l.strip()])

            result = Lean4ProofResult(
                theorem_id=theorem_id,
                theorem_name=theorem_name,
                theorem_statement=statement,
                informal_proof=informal[:1500],
                lean4_proof=lean4_proof,
                syntax_valid=syntax_valid,
                lean_verified=lean_verified,
                proof_length=n_lines,
                quality_score=quality,
                tactics_used=tactics,
                error_message=error_msg,
            )

            # Save as training pair
            training_record = {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are ProofCoach, a Lean 4 proof writer. Convert informal math proofs to formal Lean 4 code.",
                    },
                    {
                        "role": "user",
                        "content": f"Write a Lean 4 proof for:\n\n{statement}\n\nInformal approach: {informal[:800]}",
                    },
                    {
                        "role": "assistant",
                        "content": f"```lean4\n{lean4_proof}\n```",
                    },
                ],
                "metadata": asdict(result),
            }

            async with aiofiles.open(output_file, "a") as f:
                await f.write(json.dumps(training_record) + "\n")

            return result

    def _load_theorems(self) -> list[dict]:
        """Load theorem-proof pairs from input directory."""
        theorems = []
        for jsonl_file in self.input_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            t = json.loads(line)
                            # Only keep theorems with statements
                            if t.get("statement") and len(t["statement"]) > 20:
                                theorems.append(t)
                        except json.JSONDecodeError:
                            continue
        logger.info(f"Loaded {len(theorems)} theorems from {self.input_dir}")
        return theorems

    async def synthesize_all(self) -> int:
        """Synthesize Lean 4 proofs for all theorems."""
        theorems = self._load_theorems()
        output_file = self.output_dir / "lean4_proofs.jsonl"

        logger.info(f"Synthesizing Lean 4 proofs for {len(theorems)} theorems...")
        start = time.time()

        if self.backend == "vllm":
            async with aiohttp.ClientSession() as session:
                tasks = [self._synthesize_one(t, output_file, session) for t in theorems]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [self._synthesize_one(t, output_file) for t in theorems]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start
        succeeded = sum(1 for r in results if isinstance(r, Lean4ProofResult))

        logger.success(
            f"Lean4 synthesis complete in {elapsed/60:.1f}m:\n"
            f"  {self._stats['attempted']} attempted\n"
            f"  {self._stats['syntax_valid']} syntax valid\n"
            f"  {self._stats['lean_verified']} Lean verified\n"
            f"  {self._stats['quality_filtered']} quality filtered\n"
            f"  {self._stats['failed']} failed"
        )
        return succeeded


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Synthesize Lean 4 formal proofs")
    parser.add_argument("--input-dir", default="data/raw/lean4")
    parser.add_argument("--output-dir", default="data/synthesized")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    synthesizer = Lean4Synthesizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
    )
    n = asyncio.run(synthesizer.synthesize_all())
    print(f"\nTotal Lean 4 proofs synthesized: {n:,}")
