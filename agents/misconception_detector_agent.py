"""
Misconception Detector Agent — Diagnoses WHERE a student's reasoning went wrong

Not just "that's incorrect" but "I see you're assuming the function is
continuous — can you think of a case where that fails?"

Uses the trained misconception detection capability from the model
plus pattern matching for common error types.
"""

import re
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MisconceptionDiagnosis:
    """Diagnosis of a student's misconception."""

    misconception_type: str
    description: str
    corrective_question: str
    correct_answer: Optional[str]
    student_answer: Optional[str]
    confidence: float


MISCONCEPTION_PATTERNS = [
    {
        "type": "divided_by_zero",
        "pattern": r"divid\w+\s+by\s+\(?n\)?|/(n|x|y)\b",
        "description": "You divided by a variable — could this variable be zero?",
        "question": "What happens to your step if {var} = 0?",
    },
    {
        "type": "off_by_one",
        "pattern": r"first\s+n|n\s+terms|count.*from\s+0|count.*from\s+1",
        "description": "This looks like an off-by-one error in counting.",
        "question": "Let's check a small case: if n=1, does your formula give the right answer?",
    },
    {
        "type": "assumed_independence",
        "pattern": r"independent|probability\s+of\s+both|p\(a\)\s*\*\s*p\(b\)",
        "description": "You assumed independence — is this justified here?",
        "question": "Are these two events actually independent? What would make them dependent?",
    },
    {
        "type": "assumed_continuity",
        "pattern": r"continuous|differentiable|smooth",
        "description": "You assumed the function is continuous — is this given?",
        "question": "What conditions would make this function NOT continuous?",
    },
]

SYSTEM_PROMPT = """You are ProofCoach's misconception detector.

Given a student's wrong solution, diagnose the EXACT mathematical error.

Your diagnosis must:
1. Name the specific error type
2. Explain precisely where the reasoning breaks down
3. Ask a targeted Socratic question that helps the student see the error themselves

Do NOT just say "that's wrong." Pinpoint the exact logical or mathematical mistake."""


class MisconceptionDetectorAgent:
    """
    Detects and diagnoses student misconceptions.

    Two-stage detection:
    1. Pattern matching for common error types (fast, no model inference)
    2. Model-based detection for complex/novel misconceptions
    """

    def __init__(
        self,
        model_path: str = "checkpoints/proofcoach-final",
        device: str = "auto",
        use_pattern_matching: bool = True,
    ) -> None:
        self.use_pattern_matching = use_pattern_matching

        logger.info("Loading misconception detection model...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except (ValueError, ImportError):
            logger.warning(
                "flash_attention_2 not available; falling back to eager attention"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="eager",
                trust_remote_code=True,
            )
        self._model.eval()
        logger.info("MisconceptionDetectorAgent ready")

    def diagnose(
        self,
        problem: str,
        student_work: str,
        student_answer: Optional[str] = None,
        correct_answer: Optional[str] = None,
    ) -> MisconceptionDiagnosis:
        """
        Diagnose the misconception in a student's wrong solution.

        Args:
            problem: The problem statement
            student_work: The student's solution attempt
            student_answer: The answer the student got
            correct_answer: The correct answer

        Returns:
            MisconceptionDiagnosis with type, description, and corrective question
        """
        # Stage 1: pattern matching (fast)
        if self.use_pattern_matching:
            pattern_result = self._pattern_match(student_work)
            if pattern_result and pattern_result.confidence > 0.8:
                return pattern_result

        # Stage 2: model-based detection
        return self._model_detect(problem, student_work, student_answer, correct_answer)

    def _pattern_match(self, student_work: str) -> Optional[MisconceptionDiagnosis]:
        """Fast pattern-matching for common misconceptions."""
        work_lower = student_work.lower()

        for pattern in MISCONCEPTION_PATTERNS:
            if re.search(pattern["pattern"], work_lower, re.IGNORECASE):
                # Find the variable that was divided by (if relevant)
                var_match = re.search(r"/(n|x|y|k)\b", work_lower)
                var = var_match.group(1) if var_match else "the variable"

                question = pattern["question"].format(var=var)

                return MisconceptionDiagnosis(
                    misconception_type=pattern["type"],
                    description=pattern["description"],
                    corrective_question=question,
                    correct_answer=None,
                    student_answer=None,
                    confidence=0.85,
                )

        return None

    def _model_detect(
        self,
        problem: str,
        student_work: str,
        student_answer: Optional[str],
        correct_answer: Optional[str],
    ) -> MisconceptionDiagnosis:
        """Use the trained model for complex misconception detection."""
        user_content = f"Problem: {problem}\n\nStudent's approach: {student_work}\n"
        if student_answer:
            user_content += f"Student's answer: {student_answer}\n"
        if correct_answer:
            user_content += f"Correct answer: {correct_answer}\n"
        user_content += "\nDiagnose the misconception."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,  # Low temperature for focused diagnosis
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._parse_diagnosis(response, student_answer, correct_answer)

    def _parse_diagnosis(
        self,
        response: str,
        student_answer: Optional[str],
        correct_answer: Optional[str],
    ) -> MisconceptionDiagnosis:
        """Parse the model's diagnosis into a structured result."""
        # Extract the corrective question (first "?" sentence)
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        corrective_question = next(
            (s for s in sentences if s.endswith("?")),
            "Can you identify where your reasoning might break down?",
        )

        # Detect misconception type from response
        misconception_type = "unknown"
        type_keywords = {
            "zero": "divided_by_zero",
            "edge case": "missed_edge_case",
            "continuity": "assumed_continuity",
            "independence": "assumed_independence",
            "off by one": "off_by_one",
            "double count": "double_counting",
            "sufficient": "necessary_vs_sufficient",
            "if and only": "if_vs_iff_confusion",
        }
        response_lower = response.lower()
        for keyword, mtype in type_keywords.items():
            if keyword in response_lower:
                misconception_type = mtype
                break

        return MisconceptionDiagnosis(
            misconception_type=misconception_type,
            description=response[:200],
            corrective_question=corrective_question,
            correct_answer=correct_answer,
            student_answer=student_answer,
            confidence=0.75,
        )
