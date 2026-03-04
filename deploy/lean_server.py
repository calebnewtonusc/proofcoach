"""
Lean 4 Verification Microservice

A separate FastAPI service that handles Lean 4 proof verification requests.
Runs as an independent service so it doesn't block the main API server.

Endpoint:
  POST /verify    — verify a Lean 4 proposition
  GET  /health

Usage:
  python deploy/lean_server.py
  # or via docker-compose
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from core.lean4_interface import Lean4Interface, VerificationResult


app = FastAPI(title="ProofCoach Lean 4 Verification Service")

# Lazy-initialised so the module can be imported even when Lean is not installed.
# The interface is created on first request rather than at import time.
_lean4: Optional[Lean4Interface] = None


def _get_lean4() -> Lean4Interface:
    global _lean4
    if _lean4 is None:
        _lean4 = Lean4Interface(timeout=15)
    return _lean4


class VerifyRequest(BaseModel):
    theorem: str
    tactic: Optional[str] = None


class VerifyResponse(BaseModel):
    success: bool
    message: str
    elapsed_ms: float
    reward: float


@app.get("/health")
async def health():
    return {"status": "OK", "lean4": "ready"}


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    result: VerificationResult = _get_lean4().verify(req.theorem, tactic=req.tactic)
    return VerifyResponse(
        success=result.success,
        message=result.message,
        elapsed_ms=result.elapsed_ms,
        reward=result.reward,
    )


@app.post("/verify_batch")
async def verify_batch(theorems: list[str]):
    results = _get_lean4().verify_batch(theorems)
    return [
        {"success": r.success, "theorem": r.theorem, "reward": r.reward}
        for r in results
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
