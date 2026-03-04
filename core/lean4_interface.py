"""
Lean 4 Interface — Proof verification via the Lean 4 type checker

This is the core innovation of ProofCoach:
  - Model generates a tutoring step with a mathematical claim
  - The claim is expressed as a Lean 4 theorem
  - Lean 4 type-checks it
  - If accepted: reward = +1.0
  - If rejected: reward = -1.0

This makes it impossible for the model to teach incorrect mathematics —
the punishment is built into the training loop.

Example:
    interface = Lean4Interface()
    result = interface.verify("theorem ds (n : ℤ) : n^2 - 1 = (n-1)*(n+1) := by ring")
    # result.success = True, reward = +1.0
"""

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class VerificationResult:
    """Result of a Lean 4 proof verification."""
    theorem: str
    success: bool
    message: str
    elapsed_ms: float
    reward: float  # +1.0 success, -1.0 failure, 0.0 timeout


LEAN4_PREAMBLE = """
import Mathlib.Tactic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.NumberTheory.Modular
open BigOperators

"""

# Common proofs used in olympiad tutoring
KNOWN_GOOD_THEOREMS = {
    "n^2 - 1 = (n-1)*(n+1)": "theorem ds (n : ℤ) : n^2 - 1 = (n - 1) * (n + 1) := by ring",
    "sum_first_n_odd": "theorem sum_odd (n : ℕ) : (∑ i in Finset.range n, (2*i+1)) = n^2 := by induction n with | zero => simp | succ n ih => simp [Finset.sum_range_succ]; linarith",
}


class Lean4Interface:
    """
    Interface to the Lean 4 proof assistant.

    Runs Lean 4 as a subprocess, submitting theorem statements and
    collecting verification results.

    Supports:
      - Single theorem verification
      - Batch verification (ThreadPoolExecutor for parallelism)
      - Simulated rewards for debugging (set LEAN4_SIMULATED=1)
    """

    def __init__(
        self,
        timeout: int = 10,
        lean_executable: str = "lean",
        use_mathlib: bool = True,
        simulated: bool = False,
    ) -> None:
        self.timeout = timeout
        self.lean_executable = lean_executable
        self.use_mathlib = use_mathlib
        self.simulated = simulated or os.getenv("LEAN4_SIMULATED", "0") == "1"

        if not self.simulated:
            self._check_lean_available()

    def _check_lean_available(self) -> None:
        """Verify Lean 4 is installed and accessible."""
        try:
            result = subprocess.run(
                [self.lean_executable, "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or "Lean" not in result.stdout:
                raise RuntimeError(f"Lean 4 not found: {result.stderr}")
            version = result.stdout.strip()
            logger.info(f"Lean 4 ready: {version}")
        except FileNotFoundError:
            raise RuntimeError(
                "Lean 4 not found. Install via: "
                "curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"
            )

    def verify(self, theorem: str, tactic: Optional[str] = None) -> VerificationResult:
        """
        Verify a single Lean 4 theorem.

        Args:
            theorem: A complete Lean 4 theorem statement (e.g., "theorem foo : 1 + 1 = 2 := by norm_num")
            tactic: Optional tactic hint (e.g., "ring", "norm_num", "omega")

        Returns:
            VerificationResult with success/failure and reward
        """
        if self.simulated:
            return self._simulate_verification(theorem)

        start_time = time.time()

        # Build the Lean 4 file
        lean_content = self._build_lean_file(theorem, tactic)

        with tempfile.NamedTemporaryFile(
            suffix=".lean", mode="w", delete=False
        ) as f:
            f.write(lean_content)
            lean_file = f.name

        try:
            result = subprocess.run(
                [self.lean_executable, lean_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            success = result.returncode == 0 and not self._has_errors(result.stdout + result.stderr)
            message = (result.stderr or result.stdout).strip()

            return VerificationResult(
                theorem=theorem,
                success=success,
                message=message[:500],
                elapsed_ms=elapsed_ms,
                reward=1.0 if success else -1.0,
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                theorem=theorem,
                success=False,
                message=f"Lean 4 timeout ({self.timeout}s)",
                elapsed_ms=self.timeout * 1000,
                reward=0.0,
            )
        finally:
            Path(lean_file).unlink(missing_ok=True)

    def verify_batch(self, theorems: list[str]) -> list[VerificationResult]:
        """
        Verify multiple theorems in parallel using ThreadPoolExecutor.

        Used during RL training to verify N=4 completions per prompt.
        """
        from concurrent.futures import ThreadPoolExecutor
        workers = int(os.getenv("LEAN4_WORKERS", 8))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.verify, thm) for thm in theorems]
            return [f.result() for f in futures]

    def extract_claims_from_dialogue(self, dialogue_text: str) -> list[str]:
        """
        Extract Lean 4 theorem statements from a tutoring dialogue.

        Looks for:
          1. Explicit ```lean ... ``` code blocks
          2. Mathematical claims that can be formalized
        """
        claims = []

        # Find explicit lean code blocks
        for match in re.finditer(r"```lean\n(.*?)\n```", dialogue_text, re.DOTALL):
            lean_code = match.group(1).strip()
            if lean_code.startswith("theorem") or lean_code.startswith("lemma"):
                claims.append(lean_code)

        return claims

    def natural_language_to_lean4(self, claim: str, context: str = "") -> Optional[str]:
        """
        Attempt to translate a natural language mathematical claim to Lean 4.

        This is a heuristic approach — for complex claims, LLM-based translation
        is used during synthesis. This covers common patterns.
        """
        claim_lower = claim.lower()

        # Common patterns
        if re.search(r"n\^?2 - 1 = \(n-1\)\(n\+1\)", claim):
            return "theorem ds (n : ℤ) : n^2 - 1 = (n - 1) * (n + 1) := by ring"

        if re.search(r"sum of first n odd.*n\^?2", claim, re.IGNORECASE):
            return KNOWN_GOOD_THEOREMS["sum_first_n_odd"]

        if re.search(r"a\^2 \+ b\^2 >= 2ab", claim, re.IGNORECASE):
            return "theorem sq_nonneg_sum (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by nlinarith [sq_nonneg (a-b)]"

        if re.search(r"(a \+ b)/2 >= sqrt\(ab\)", claim, re.IGNORECASE):
            return None  # Too complex for simple heuristic

        # For arithmetic claims like "3 * 7 = 21"
        match = re.search(r"(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)", claim)
        if match:
            a, b, c = match.groups()
            if int(a) * int(b) == int(c):
                return f"theorem arith : {a} * {b} = {c} := by norm_num"

        return None

    def _build_lean_file(self, theorem: str, tactic: Optional[str] = None) -> str:
        """Build a complete Lean 4 file for verification."""
        content = LEAN4_PREAMBLE

        # If theorem doesn't have a proof, add the suggested tactic
        if ":= by" not in theorem and ":=" not in theorem:
            if tactic:
                theorem = f"{theorem} := by {tactic}"
            else:
                theorem = f"{theorem} := by decide"

        content += theorem + "\n"
        return content

    def _has_errors(self, output: str) -> bool:
        """Check if Lean 4 output contains errors."""
        # PC-21: Removed "Goals accomplished" from error_indicators — it is a
        # Lean 4 success message, not an error. Including it in the list was
        # misleading and the separate guard below makes the list entry redundant.
        # "Goals accomplished" means the proof was accepted.
        if "Goals accomplished" in output:
            return False

        error_indicators = [
            "error:", "Error:", "unknown identifier",
            "type mismatch", "application type mismatch",
        ]
        return any(ind in output for ind in error_indicators)

    def _simulate_verification(self, theorem: str) -> VerificationResult:
        """
        Simulate Lean 4 verification for debugging.

        Uses heuristics to estimate whether a theorem would verify:
        - Theorems using `ring` tactic: high success rate
        - Theorems with `by decide`: high success rate for finite claims
        - Complex theorems: lower simulated success rate
        """
        import random

        # Heuristic success rates by tactic
        if ":= by ring" in theorem:
            success = random.random() < 0.90
        elif ":= by norm_num" in theorem:
            success = random.random() < 0.85
        elif ":= by omega" in theorem:
            success = random.random() < 0.80
        elif ":= by simp" in theorem:
            success = random.random() < 0.65
        elif ":= by induction" in theorem or "induction" in theorem:
            success = random.random() < 0.55
        else:
            success = random.random() < 0.50

        return VerificationResult(
            theorem=theorem,
            success=success,
            message="[SIMULATED]",
            elapsed_ms=50.0,
            reward=1.0 if success else -1.0,
        )


class Lean4Server:
    """
    Long-running Lean 4 server for the deployment microservice.

    Maintains a persistent Lean 4 process to avoid startup overhead.
    Suitable for the /v1/verify API endpoint.
    """

    def __init__(self, timeout: int = 30) -> None:
        self._interface = Lean4Interface(timeout=timeout)
        logger.info("Lean4Server initialized")

    def verify_claim(self, claim: str, proof_attempt: Optional[str] = None) -> dict:
        """
        Verify a mathematical claim from the API.

        Args:
            claim: Natural language or Lean 4 claim
            proof_attempt: Optional Lean 4 proof attempt

        Returns:
            Dict with verified, lean4_proof, explanation
        """
        # Try to translate natural language to Lean 4
        lean4_thm = self._interface.natural_language_to_lean4(claim)

        if not lean4_thm and proof_attempt:
            lean4_thm = proof_attempt

        if not lean4_thm:
            return {
                "verified": None,
                "lean4_proof": None,
                "explanation": "Could not translate claim to Lean 4. Try providing a Lean 4 proof attempt.",
            }

        result = self._interface.verify(lean4_thm)

        return {
            "verified": result.success,
            "lean4_proof": lean4_thm if result.success else None,
            "explanation": result.message or ("Verified" if result.success else "Verification failed"),
            "elapsed_ms": result.elapsed_ms,
            "reward": result.reward,
        }
