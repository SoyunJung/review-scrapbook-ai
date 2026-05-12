from __future__ import annotations

from typing import Any

from src.llm import BaseLLMClient
from src.models import ReflectionInput


def generate_followup_questions(
    llm: BaseLLMClient,
    reflection_input: ReflectionInput,
) -> list[str]:
    questions = llm.generate_questions(reflection_input)
    normalized = [question.strip() for question in questions if question.strip()]
    if not normalized:
        raise ValueError("생성된 후속 질문이 없습니다.")
    return normalized[:5]


def build_structured_memo(
    llm: BaseLLMClient,
    reflection_input: ReflectionInput,
    answers: dict[str, str],
) -> dict[str, Any]:
    memo = llm.generate_memo(reflection_input, answers)
    memo.setdefault("one_line_summary", f"{reflection_input.title} 감상 기록")
    memo.setdefault("content_summary", reflection_input.initial_note)
    memo.setdefault("impressive_points", [])
    memo.setdefault("personal_interpretation", "아직 기록되지 않았습니다.")
    memo.setdefault("emotional_trace", "아직 기록되지 않았습니다.")
    memo.setdefault("revisit_points", "아직 기록되지 않았습니다.")
    memo.setdefault("comparison_points", "아직 기록되지 않았습니다.")
    memo.setdefault("review_note", reflection_input.initial_note)
    memo.setdefault("tags", [reflection_input.content_type, reflection_input.status])
    return memo
