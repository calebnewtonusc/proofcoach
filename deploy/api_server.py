"""
ProofCoach API Server

FastAPI application providing the ProofCoach tutoring, verification,
diagnosis, and sequencing endpoints.

Endpoints:
  POST /v1/tutor      — Socratic tutoring response
  POST /v1/verify     — Lean 4 proof verification
  POST /v1/diagnose   — Misconception diagnosis
  POST /v1/sequence   — Next practice problem recommendation
  GET  /v1/problems/{competition}/{year}  — List problems
  WS   /v1/stream     — WebSocket streaming tutoring
  GET  /health
"""

import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

# Initialize agents lazily to avoid slow startup in health checks
_tutor_agent = None
_verifier_agent = None
_detector_agent = None
_sequencer_agent = None


def get_tutor():
    global _tutor_agent
    if _tutor_agent is None:
        from agents.tutor_agent import TutorAgent
        _tutor_agent = TutorAgent(
            model_path=os.getenv("MODEL_PATH", "checkpoints/proofcoach-final")
        )
    return _tutor_agent


def get_verifier():
    global _verifier_agent
    if _verifier_agent is None:
        from agents.proof_verifier_agent import ProofVerifierAgent
        _verifier_agent = ProofVerifierAgent()
    return _verifier_agent


def get_detector():
    global _detector_agent
    if _detector_agent is None:
        from agents.misconception_detector_agent import MisconceptionDetectorAgent
        _detector_agent = MisconceptionDetectorAgent(
            model_path=os.getenv("MODEL_PATH", "checkpoints/proofcoach-final")
        )
    return _detector_agent


def get_sequencer():
    global _sequencer_agent
    if _sequencer_agent is None:
        from agents.practice_sequencer_agent import PracticeSequencerAgent
        _sequencer_agent = PracticeSequencerAgent(
            problem_bank_dir=os.getenv("RAW_DATA_DIR", "data/raw")
        )
    return _sequencer_agent


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class TutorRequest(BaseModel):
    problem: str
    student_work: Optional[str] = None
    session_id: str
    student_id: Optional[str] = None
    hint_level: Optional[int] = None


class TutorResponse(BaseModel):
    question: str
    hint_level: int
    verified_steps: list[str]
    approach_hint: Optional[str] = None
    next_problem_suggestion: Optional[str] = None
    lean4_claims: list[dict]
    session_id: str


class VerifyRequest(BaseModel):
    claim: str
    proof_attempt: Optional[str] = None


class VerifyResponse(BaseModel):
    verified: Optional[bool]
    lean4_proof: Optional[str] = None
    explanation: str
    elapsed_ms: float


class DiagnoseRequest(BaseModel):
    problem: str
    student_work: str
    student_answer: Optional[str] = None
    correct_answer: Optional[str] = None


class DiagnoseResponse(BaseModel):
    correct: bool
    misconception: Optional[str] = None
    misconception_type: Optional[str] = None
    corrective_question: Optional[str] = None
    correct_answer: Optional[str] = None


class SequenceRequest(BaseModel):
    student_id: str
    session_history: list[str] = []


class SequenceResponse(BaseModel):
    next_problem: Optional[dict] = None
    reason: str
    target_skill_gap: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ProofCoach API",
    description="Socratic math tutoring with Lean 4 verified proof steps",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "OK", "timestamp": time.time()}


@app.post("/v1/tutor", response_model=TutorResponse)
async def tutor(req: TutorRequest):
    """
    Generate a Socratic tutoring response.

    Given a problem and the student's current work, returns a targeted
    question or hint that guides the student toward the insight without
    giving away the answer.
    """
    try:
        agent = get_tutor()
        result = agent.tutor(
            problem=req.problem,
            student_work=req.student_work,
            session_id=req.session_id,
            student_id=req.student_id,
            force_hint_level=req.hint_level,
        )
        return TutorResponse(**{k: result[k] for k in TutorResponse.model_fields})
    except Exception as e:
        logger.error(f"Tutor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """
    Verify a mathematical claim using Lean 4.

    Accepts either a natural language claim or a Lean 4 theorem statement.
    Returns whether Lean 4 accepts the claim and the formal proof if successful.
    """
    try:
        agent = get_verifier()
        result = agent.verify_proof(claim=req.claim, proof_attempt=req.proof_attempt)
        return VerifyResponse(
            verified=result.verified,
            lean4_proof=result.lean4_proof,
            explanation=result.explanation,
            elapsed_ms=result.elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest):
    """
    Diagnose a student misconception.

    Given a problem and a student's incorrect work, identifies the specific
    error and returns a corrective Socratic question.
    """
    try:
        agent = get_detector()
        result = agent.diagnose(
            problem=req.problem,
            student_work=req.student_work,
            student_answer=req.student_answer,
            correct_answer=req.correct_answer,
        )

        correct = (
            req.student_answer == req.correct_answer
            if req.student_answer and req.correct_answer
            else False
        )

        return DiagnoseResponse(
            correct=correct,
            misconception=result.description,
            misconception_type=result.misconception_type,
            corrective_question=result.corrective_question,
            correct_answer=req.correct_answer,
        )
    except Exception as e:
        logger.error(f"Diagnose error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/sequence", response_model=SequenceResponse)
async def sequence(req: SequenceRequest):
    """
    Get the next recommended practice problem for a student.

    Uses the student's skill model to identify the weakest area
    and select a problem at the appropriate difficulty.
    """
    try:
        agent = get_sequencer()
        recommendation = agent.get_next_problem(
            student_id=req.student_id,
            session_history=req.session_history,
        )

        if recommendation is None:
            return SequenceResponse(
                next_problem=None,
                reason="No suitable problem found for current skill level.",
            )

        return SequenceResponse(
            next_problem={
                "problem_id": recommendation.problem_id,
                "competition": recommendation.competition,
                "year": recommendation.year,
                "number": recommendation.number,
                "statement": recommendation.statement,
                "difficulty": recommendation.difficulty,
                "topics": recommendation.topics,
            },
            reason=recommendation.reason,
            target_skill_gap=recommendation.target_skill_gap,
        )
    except Exception as e:
        logger.error(f"Sequence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket):
    """WebSocket endpoint for streaming tutoring responses."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            problem = data.get("problem", "")
            student_work = data.get("student_work", "")
            session_id = data.get("session_id", "ws_session")

            # For streaming, we chunk the response
            agent = get_tutor()
            result = agent.tutor(
                problem=problem,
                student_work=student_work,
                session_id=session_id,
            )
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "deploy.api_server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info",
    )
