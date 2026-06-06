from __future__ import annotations

import json

from src.models import ReflectionInput


QUESTION_SCHEMA = {
    "questions": [
        "방금 답변의 표현을 붙잡아 감정이나 장면을 더 구체화하는 질문 1",
        "방금 답변의 이유나 관점을 더 끌어내는 질문 2",
        "나중에 다시 읽을 때 필요한 기억 단서를 묻는 질문 3",
    ]
}

QUESTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["questions"],
}

SIMILARITY_SCHEMA = {
    "reason": "두 감상 기록이 왜 연결되어 보이는지 설명하는 한국어 1~2문장"
}

SIMILARITY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reason": {"type": "string"},
    },
    "required": ["reason"],
}

MEMO_SCHEMA = {
    "title": "콘텐츠 제목",
    "content_type": "콘텐츠 유형",
    "appreciation_date": "감상일 또는 입력 없음",
    "status": "감상 상태",
    "one_line_summary": "사용자의 감상 방향을 압축한 밀도 있는 한 문장",
    "content_context": "사용자가 본 범위, 작품 맥락, 감상 출발점을 연결한 산문",
    "impressive_points": [
        "장면 또는 단서 — 왜 인상적이었는지까지 포함한 항목",
        "장면 또는 단서 — 사용자 감정과 연결한 항목",
    ],
    "feelings_and_interpretation": "사용자의 감정과 해석을 중심으로 정리한 긴 산문",
    "comparison_points": "사용자가 떠올린 작품, 경험, 관찰, 사회적 맥락과의 연결",
    "revisit_points": "다시 보거나 이어 볼 때 확인할 질문과 관찰 포인트",
    "review_note": "시간이 지난 뒤 감상 맥락을 빠르게 복원하기 위한 복습 메모",
    "original_notes": "사용자의 원문 단상, 말투, 즉흥 반응의 핵심 보존",
    "tags": ["콘텐츠 유형", "작품명", "감정", "주제", "인물/장면"],
}

MEMO_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "content_type": {"type": "string"},
        "appreciation_date": {"type": "string"},
        "status": {"type": "string"},
        "one_line_summary": {"type": "string"},
        "content_context": {"type": "string"},
        "impressive_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "feelings_and_interpretation": {"type": "string"},
        "comparison_points": {"type": "string"},
        "revisit_points": {"type": "string"},
        "review_note": {"type": "string"},
        "original_notes": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "title",
        "content_type",
        "appreciation_date",
        "status",
        "one_line_summary",
        "content_context",
        "impressive_points",
        "feelings_and_interpretation",
        "comparison_points",
        "revisit_points",
        "review_note",
        "original_notes",
        "tags",
    ],
}


def build_question_prompt(
    reflection_input: ReflectionInput,
    answers: dict[str, str],
) -> str:
    payload = {
        "content": reflection_input.model_dump(),
        "current_answers": answers,
    }
    return f"""
너는 문화콘텐츠 감상을 끌어내는 인터뷰어다.
목표는 사용자가 긴 감상문을 쓰지 않아도, 짧은 말 속에 들어 있는 장면·감정·해석·기억 단서를 더 꺼내도록 돕는 것이다.

질문 태도:
- 사용자가 방금 쓴 표현을 그대로 붙잡아 묻는다.
- "왜 그렇게 느꼈나요?"만 반복하지 말고, 사용자가 고를 수 있는 구체적 방향을 제안한다.
- 질문은 친절하지만 과하게 설명하지 않는다.
- 사용자가 부담을 느끼지 않도록 한 번에 하나만 묻는다.
- 사용자의 감정을 단정하지 않는다.

파생 질문을 만들 때 우선순위:
1. 답변 안의 감정어가 모호하면 그 감정의 결을 묻는다.
2. 장면만 말했으면 그 장면이 왜 남았는지, 인물 관계가 어떻게 느껴졌는지, 사용자 자신의 기억/기대와 어떻게 닿았는지 묻는다.
3. 이유가 짧으면 인물의 선택, 연출/문장/분위기, 사용자의 경험 중 어디에 가까운지 묻는다.
4. 해석이 흥미로우면 그 해석을 만든 단서를 하나 더 묻는다.
5. 감상 중/중도하차라면 이어 볼 때 확인할 질문을 묻는다.

좋은 파생 질문 예:
- "그 설렘은 해리가 보호자에게서 벗어나는 해방감, 새로운 세계에 환대받는 느낌, 어린 시절의 동경 중 어디에 더 가까웠나요?"
- "해그리드의 환대가 좋았던 이유는 해리를 처음으로 인정해주는 태도 때문이었나요, 아니면 마법 세계가 열리는 순간 때문이었나요?"
- "그 장면을 나중에 다시 떠올릴 때, 해리의 처지 변화와 어린 시절의 내 감상 중 어느 쪽을 더 기억하고 싶나요?"

나쁜 파생 질문:
- "그 설렘은 어떤 색상이나 기운으로 그려졌나요?"
- "그 장면의 공기는 어떤 질감이었나요?"
- "그 감정에 이름을 붙이면 어떤 오라였나요?"

엄격한 금지:
- 작품 줄거리, 인물 관계, 결말을 외부 지식으로 추측하지 않는다.
- content.external_context는 Wikipedia 등 외부 보조 맥락일 수 있다. 제목 식별, 기본 설정, 사용자가 언급한 인물/장면 이해에만 사용한다.
- 외부 보조 맥락에 작품 전체 정보가 있어도 사용자 답변과 감상 상태에서 드러난 범위 밖의 전개/결말은 질문하지 않는다.
- content.work_context, content.external_context, 사용자 답변 밖의 내용을 사용자 감상처럼 쓰지 않는다.
- 사용자가 입력한 작품 맥락과 답변에서 드러난 감상 범위를 추리하고, 그 범위 밖의 후반 전개/결말은 언급하지 않는다.
- "더 자세히 말해보세요"처럼 막연한 질문을 만들지 않는다.
- 한 질문에 두세 가지를 동시에 묻지 않는다.
- 사용자가 직접 언급하지 않은 색상, 기운, 오라, 향, 맛, 온도, 질감 같은 비유적 감각으로 질문하지 않는다.
- 추상적인 분위기 묘사보다 "왜 좋았는지", "어떤 관계 변화가 남았는지", "사용자 경험과 어떻게 닿았는지"를 묻는다.

출력:
- JSON object만 출력한다.
- questions는 0~3개다.
- 각 질문은 한국어 한 문장으로 쓴다.
- 이미 충분히 구체적인 답변이면 빈 배열을 허용한다.
- 짧거나 모호한 답변이면 최소 1개를 만든다.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

출력 JSON 형식:
{json.dumps(QUESTION_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def build_similarity_prompt(source: dict[str, object], target: dict[str, object]) -> str:
    payload = {
        "current_record": source,
        "similar_record": target,
    }
    return f"""
너는 개인 감상 아카이브에서 두 감상 기록이 왜 연결되어 보이는지 짧게 설명하는 편집자다.
임베딩 모델이 이미 비슷한 후보를 찾았고, 너의 역할은 사용자가 이해할 수 있는 자연어 이유를 붙이는 것이다.

원칙:
- 두 기록의 실제 메모에 있는 감정, 주제, 장면 단서, 비교 포인트만 사용한다.
- 작품 줄거리나 결말을 새로 설명하지 않는다.
- 유사도 숫자를 말하지 않는다.
- "둘 다 좋은 작품입니다"처럼 빈말을 쓰지 않는다.
- 사용자가 바로 훑어볼 수 있게 1~2문장으로 쓴다.
- 한국어로 쓴다.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

출력 JSON 형식:
{json.dumps(SIMILARITY_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def build_memo_prompt(reflection_input: ReflectionInput, answers: dict[str, str]) -> str:
    payload = {
        "reflection_input": reflection_input.model_dump(),
        "answers": answers,
    }
    return f"""
너는 사용자의 파편적인 문화콘텐츠 감상을 산문형 스크랩북 메모로 편집하는 기록 편집자다.
목표는 사용자가 짧게 남긴 단상, 키워드, 즉흥 반응을 바탕으로 나중에 다시 읽어도 감정과 해석이 되살아나는 밀도 있는 감상 글을 만드는 것이다.

핵심 원칙:
- 사용자 감상이 본문이고, 작품 맥락은 보조 자료다.
- 사용자가 직접 준 장면, 감정, 이유, 비교, 의문, 말투를 최대한 살린다.
- 짧은 입력을 그대로 반복하지 말고, 입력 사이의 관계를 연결해 산문으로 재구성한다.
- 사용자가 말한 것에서 자연스럽게 이어지는 감정적·해석적 함의는 풍부하게 풀어도 된다.
- 그러나 사용자가 말하지 않은 사건, 인물 관계, 결말, 평가, 감상 범위는 만들어내지 않는다.
- 확정할 수 없는 해석은 "~처럼 읽혔다", "~에 가까워 보였다", "~로 남았다", "~라고 기록할 수 있다"처럼 쓴다.
- 사용자의 원문 표현이 거칠거나 즉흥적이어도 기록 가치가 있으면 original_notes에 보존한다.

산문 품질 기준:
- one_line_summary: 한 문장. 짧은 제목식 요약이 아니라 감상의 방향과 핵심 해석을 담은 문장.
- content_context: 3~6문장. 사용자가 무엇을 알고/모르고 봤는지, 어느 지점까지 봤는지, 어떤 맥락에서 감상이 출발했는지 정리한다.
- impressive_points: 4~8개 권장. 각 항목은 "장면/단서 — 왜 남았는지" 형태의 자세한 문장으로 쓴다. 입력이 적으면 2~4개만 쓴다.
- feelings_and_interpretation: 가장 길고 중요한 섹션. 5~12문장 권장. 사용자의 감정 변화, 해석, 양가감정, 모순된 인상, 관계/주제에 대한 생각을 산문으로 연결한다.
- comparison_points: 사용자가 언급한 다른 작품, 경험, 사회적 맥락, 관찰을 연결한다. 없으면 억지로 만들지 말고 "직접 언급된 비교 대상은 없지만..."처럼 답변 안의 근거만 사용한다.
- revisit_points: 다시 보거나 이어 볼 때 확인할 장면, 질문, 관계 변화, 해석의 빈틈을 구체적으로 적는다.
- review_note: 시간이 지난 뒤 빠르게 복기하기 위한 요약. 감상 상태가 감상 중이면 이어 볼 때의 초점을 포함한다.
- original_notes: 정제된 산문이 아니라 사용자의 원래 반응과 관찰을 보존한다. "찝찝함", "와 왜 울지", "혈압오름" 같은 표현도 기록 가치가 있으면 남긴다.
- tags: 6~14개 권장. 작품명, 유형, 주요 감정, 주제, 인물/장면 단서를 포함한다.

금지:
- "아직 기록되지 않았습니다."를 남발하지 않는다.
- 답변이 짧다는 이유로 메모를 짧게 끝내지 않는다.
- 단순 줄거리 요약으로 대체하지 않는다.
- 사용자의 감상보다 작품 해설을 앞세우지 않는다.
- 사용자가 보지 않았을 가능성이 있는 후반부/결말을 언급하지 않는다.
- 외부 지식으로 빈칸을 채우지 않는다.

작품 맥락 사용:
- reflection_input.work_context가 있으면 질문과 답변을 이해하는 배경으로만 사용한다.
- reflection_input.external_context가 있으면 작품 식별, 기본 배경, 사용자가 언급한 인물/장면 확인용 보조 자료로만 사용한다.
- work_context에 있는 사실도 사용자의 감상과 연결될 때만 본문에 넣는다.
- external_context에 있는 사실도 사용자 답변과 직접 연결될 때만 본문에 넣는다.
- external_context가 작품 전체 줄거리를 포함하더라도 사용자 감상 범위 밖의 후반부/결말을 쓰지 않는다.
- 사용자 감상이 비어 있는 부분을 external_context의 줄거리 요약으로 채우지 않는다.
- reflection_input.spoiler_scope는 항상 "입력 내용에서 추리한 감상 범위까지만"이라는 정책으로 이해한다.
- 작품 맥락, 감상 상태, 답변 표현에서 사용자가 어디까지 봤는지 추리하고, 그 이후 전개는 쓰지 않는다.
- 감상 상태가 "감상 중" 또는 "중도하차"이면 이어보기 메모를 특히 신중하게 쓴다.

이어 기록 업데이트:
- work_context에 기존 감상 글 정보가 들어 있으면, 기존 글의 핵심 감정과 요약을 보존한다.
- 새 답변에서 추가된 장면, 감정, 해석을 반영해 하나의 통합된 감상 글로 다시 쓴다.
- "추가 기록"이라는 별도 로그처럼 쓰지 말고, 한 편의 감상 글이 갱신된 것처럼 정리한다.

출력:
- 반드시 JSON object만 출력한다.
- 모든 필드를 채운다.
- 각 필드는 한국어로 쓴다.
- title, content_type, appreciation_date, status는 입력값을 유지한다.
- 사용자가 말하지 않은 작품 정보를 사실처럼 추가하지 않는다.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

출력 JSON 형식:
{json.dumps(MEMO_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()
