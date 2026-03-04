"""
Proof Verifier Agent — Lean 4 verification of student proof steps

Students can submit proof attempts and get formal verification.
Wraps the Lean4Interface to provide an agent-style API.

API endpoint: POST /v1/verify
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from core.lean4_interface import Lean4Interface, Lean4Server, VerificationResult


@dataclass
class ProofVerificationResponse:
    """Response from the proof verifier agent."""
    verified: bool
    lean4_proof: Optional[str]
    explanation: str
    steps_verified: list[dict]
    overall_correct: bool
    feedback: str
    elapsed_ms: float


class ProofVerifierAgent:
    """
    Agent that verifies mathematical proof steps using Lean 4.

    Provides:
    1. Step-by-step proof verification
    2. Counterexample generation for false claims
    3. Feedback on what went wrong
    """

    def __init__(
        self,
        timeout: int = 15,
        simulated: bool = False,
    ) -> None:
        self._lean4 = Lean4Interface(timeout=timeout, simulated=simulated)
        self._server = Lean4Server(timeout=timeout)
        logger.info("ProofVerifierAgent ready")

    def verify_proof(
        self,
        claim: str,
        proof_attempt: Optional[str] = None,
    ) -> ProofVerificationResponse:
        """
        Verify a single mathematical claim.

        Args:
            claim: Natural language claim or Lean 4 theorem
            proof_attempt: Optional Lean 4 proof (if not provided, tactic inference attempted)

        Returns:
            ProofVerificationResponse with verification result and feedback
        """
        result = self._server.verify_claim(claim, proof_attempt)

        feedback = self._generate_feedback(claim, result)

        return ProofVerificationResponse(
            verified=result.get("verified", False),
            lean4_proof=result.get("lean4_proof"),
            explanation=result.get("explanation", ""),
            steps_verified=[],
            overall_correct=result.get("verified", False),
            feedback=feedback,
            elapsed_ms=result.get("elapsed_ms", 0),
        )

    def verify_proof_steps(
        self,
        problem: str,
        steps: list[str],
    ) -> ProofVerificationResponse:
        """
        Verify a multi-step proof.

        Each step is verified independently. If any step fails,
        the agent identifies which step failed and why.

        Args:
            problem: The problem being proved
            steps: List of proof steps as natural language or Lean 4

        Returns:
            ProofVerificationResponse with per-step results
        """
        steps_results = []
        all_verified = True
        total_elapsed = 0.0

        for i, step in enumerate(steps):
            result = self._server.verify_claim(step)
            verified = result.get("verified", False)
            steps_results.append({
                "step": i + 1,
                "claim": step,
                "verified": verified,
                "lean4": result.get("lean4_proof"),
                "explanation": result.get("explanation", ""),
            })
            total_elapsed += result.get("elapsed_ms", 0)

            if not verified:
                all_verified = False
                # Don't stop — verify all steps to give complete feedback

        feedback = self._generate_multi_step_feedback(steps_results)

        return ProofVerificationResponse(
            verified=all_verified,
            lean4_proof=None,  # No single proof for multi-step
            explanation=f"{sum(r['verified'] for r in steps_results)}/{len(steps)} steps verified",
            steps_verified=steps_results,
            overall_correct=all_verified,
            feedback=feedback,
            elapsed_ms=total_elapsed,
        )

    def _generate_feedback(self, claim: str, result: dict) -> str:
        """Generate human-readable feedback for a verification result."""
        if result.get("verified"):
            return (
                f"Correct. Your claim '{claim[:100]}' has been formally verified by Lean 4. "
                f"Every step of this reasoning is sound."
            )
        elif result.get("verified") is None:
            return (
                f"I wasn't able to express this claim formally in Lean 4 automatically. "
                f"This doesn't mean it's wrong — try providing a Lean 4 proof attempt, "
                f"or we can discuss the reasoning informally."
            )
        else:
            explanation = result.get("explanation", "")
            return (
                f"This claim didn't verify. {explanation}\n"
                f"Double-check: is the statement exactly right? "
                f"Are there any edge cases (n=0, negative numbers) where it might fail?"
            )

    def _generate_multi_step_feedback(self, steps_results: list[dict]) -> str:
        """Generate feedback for a multi-step proof."""
        n_total = len(steps_results)
        n_verified = sum(r["verified"] for r in steps_results)

        if n_verified == n_total:
            return f"All {n_total} steps verified. Your proof is complete and formally correct."

        failed = [r for r in steps_results if not r["verified"]]
        first_fail = failed[0]

        return (
            f"{n_verified}/{n_total} steps verified. "
            f"Step {first_fail['step']} did not verify: '{first_fail['claim'][:100]}'. "
            f"{first_fail.get('explanation', '')}\n"
            f"Fix this step before the rest of the proof can proceed."
        )
