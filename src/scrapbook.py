from __future__ import annotations

from typing import Any

from src.llm import BaseLLMClient
from src.models import ReflectionInput


UNKNOWN_TEXT = "아직 기록되지 않았습니다."
VAGUE_HINTS = (
    "좋",
    "별로",
    "찝찝",
    "이상",
    "애매",
    "불편",
    "불안",
    "슬프",
    "재밌",
    "기억",
    "인상",
    "모르",
    "복잡",
    "묘",
    "강렬",
    "먹먹",
    "답답",
    "헷갈",
    "낯설",
)
FOLLOWUP_FALLBACKS = [
    "그 장면이 남은 이유는 인물의 처지 변화, 관계의 온도, 내 기억과 닿은 지점 중 어디에 더 가까웠나요?",
    "그 이유는 인물의 선택, 분위기, 내 경험 중 어디에 더 가까웠나요?",
    "그 감정은 편안함, 불편함, 아쉬움, 설렘 중 어느 쪽에 가까웠나요?",
    "떠오른 작품이나 경험과 어떤 점이 닮았나요?",
    "나중에 다시 볼 때 꼭 확인하고 싶은 장면이나 질문이 있나요?",
]
UNGROUNDED_SENSORY_TERMS = (
    "색상",
    "색깔",
    "색채",
    "기운",
    "오라",
    "향",
    "냄새",
    "맛",
    "온도",
    "질감",
)


def generate_followup_questions(
    llm: BaseLLMClient,
    reflection_input: ReflectionInput,
    answers: dict[str, str],
) -> list[str]:
    questions = llm.generate_questions(reflection_input, answers)
    answer_text = " ".join(str(answer) for answer in answers.values() if str(answer).strip())
    seen: set[str] = set()
    normalized: list[str] = []
    for question in questions:
        cleaned = str(question).strip()
        if not cleaned or cleaned in seen or not is_grounded_followup_question(cleaned, answer_text):
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized[:3]


def is_grounded_followup_question(question: str, answer_text: str) -> bool:
    question = question.strip()
    answer_text = answer_text.strip()
    if not question:
        return False
    for term in UNGROUNDED_SENSORY_TERMS:
        if term in question and term not in answer_text:
            return False
    return True


def should_suggest_followup(answer: str) -> bool:
    cleaned = answer.strip()
    if not cleaned:
        return False
    if len(cleaned) <= 180:
        return True
    sentence_count = sum(cleaned.count(mark) for mark in (".", "!", "?", "。", "！", "？"))
    if sentence_count <= 1 and len(cleaned) <= 260:
        return True
    return any(hint in cleaned for hint in VAGUE_HINTS)


def fallback_followup_question(question_index: int) -> str:
    index = min(max(question_index, 0), len(FOLLOWUP_FALLBACKS) - 1)
    return FOLLOWUP_FALLBACKS[index]


def build_structured_memo(
    llm: BaseLLMClient,
    reflection_input: ReflectionInput,
    answers: dict[str, str],
) -> dict[str, Any]:
    memo = llm.generate_memo(reflection_input, answers)
    return normalize_structured_memo(memo, reflection_input, answers)


def normalize_structured_memo(
    memo: dict[str, Any],
    reflection_input: ReflectionInput,
    answers: dict[str, str],
) -> dict[str, Any]:
    """Keep stored memo shape stable even when a provider returns loose JSON."""
    normalized = dict(memo)
    source_text = _join_nonempty(
        [
            reflection_input.work_context,
            reflection_input.initial_note,
            reflection_input.free_text,
            reflection_input.audio_transcript,
            *answers.values(),
        ]
    )

    normalized["title"] = reflection_input.title
    normalized["content_type"] = reflection_input.content_type
    normalized["appreciation_date"] = reflection_input.appreciation_date or "입력 없음"
    normalized["status"] = reflection_input.status

    defaults = {
        "one_line_summary": f"{reflection_input.title} 감상 기록",
        "content_context": reflection_input.work_context or reflection_input.initial_note or reflection_input.free_text or reflection_input.audio_transcript or UNKNOWN_TEXT,
        "feelings_and_interpretation": UNKNOWN_TEXT,
        "comparison_points": UNKNOWN_TEXT,
        "revisit_points": UNKNOWN_TEXT,
        "review_note": reflection_input.initial_note or reflection_input.free_text or reflection_input.audio_transcript or UNKNOWN_TEXT,
        "original_notes": source_text or UNKNOWN_TEXT,
    }
    for field, default in defaults.items():
        normalized[field] = _clean_text(normalized.get(field), default)

    normalized["impressive_points"] = _clean_list(
        normalized.get("impressive_points"),
        fallback=[reflection_input.initial_note or reflection_input.free_text or reflection_input.audio_transcript or UNKNOWN_TEXT],
    )
    normalized["tags"] = _clean_list(
        normalized.get("tags"),
        fallback=[reflection_input.content_type, reflection_input.status],
        max_items=14,
    )
    return normalized


def _clean_text(value: Any, fallback: str) -> str:
    if isinstance(value, list):
        value = "\n".join(str(item).strip() for item in value if str(item).strip())
    cleaned = str(value or "").strip()
    return cleaned or fallback


def _clean_list(
    value: Any,
    *,
    fallback: list[str],
    max_items: int | None = None,
) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value:
        items = str(value).splitlines()
    else:
        items = fallback

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip().lstrip("-").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if max_items and len(cleaned) >= max_items:
            break
    return cleaned or fallback


def _join_nonempty(values: list[str]) -> str:
    return "\n".join(value.strip() for value in values if value and value.strip())
