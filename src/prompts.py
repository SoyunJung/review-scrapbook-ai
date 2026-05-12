from __future__ import annotations

import json

from src.models import ReflectionInput


QUESTION_SCHEMA = {
    "questions": [
        "짧고 구체적인 후속 질문 1",
        "짧고 구체적인 후속 질문 2",
        "짧고 구체적인 후속 질문 3",
    ]
}

MEMO_SCHEMA = {
    "one_line_summary": "한 줄 요약",
    "content_summary": "줄거리 또는 핵심 내용 요약",
    "impressive_points": ["인상 깊은 포인트"],
    "personal_interpretation": "사용자의 해석",
    "emotional_trace": "감정 기록",
    "revisit_points": "다시 생각할 지점",
    "comparison_points": "비교 포인트",
    "review_note": "이어보기용 복습 메모",
    "tags": ["태그"],
}


def build_question_prompt(reflection_input: ReflectionInput) -> str:
    payload = reflection_input.model_dump()
    return f"""
너는 문화콘텐츠 감상 기록을 돕는 질문 생성자다.
목표는 사용자가 긴 감상문을 직접 쓰지 않아도 기억 복원에 필요한 단서를 꺼내도록 돕는 것이다.

규칙:
- 한국어로 답한다.
- 질문은 3~5개만 만든다.
- 질문은 짧고 답하기 쉬워야 한다.
- 사용자가 제공하지 않은 줄거리나 결말을 추측하지 않는다.
- 감상 상태가 감상 중/중도하차라면 결말을 단정하지 않는다.
- 반드시 JSON object만 출력한다.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

출력 JSON 형식:
{json.dumps(QUESTION_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def build_memo_prompt(reflection_input: ReflectionInput, answers: dict[str, str]) -> str:
    payload = {
        "reflection_input": reflection_input.model_dump(),
        "followup_answers": answers,
    }
    return f"""
너는 문화콘텐츠 감상을 구조화된 스크랩북 메모로 정리하는 기록 도우미다.
목표는 사용자가 나중에 다시 읽었을 때 당시의 감정, 해석, 기억 단서를 빠르게 복원하게 하는 것이다.

규칙:
- 한국어로 답한다.
- 사용자가 말하지 않은 사실, 결말, 평가를 지어내지 않는다.
- 문장은 자연스럽게 정리하되 사용자의 관점을 과장하지 않는다.
- 감상 중/중도하차 상태라면 아직 모르는 내용을 단정하지 않는다.
- 비어 있는 항목은 "아직 기록되지 않았습니다."처럼 솔직하게 처리한다.
- 반드시 JSON object만 출력한다.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

출력 JSON 형식:
{json.dumps(MEMO_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()
