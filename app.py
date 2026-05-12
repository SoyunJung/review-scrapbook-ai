from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.config import AppConfig
from src.db import get_record, init_db, list_records, save_record
from src.llm import LLMClientError, create_llm_client
from src.models import ReflectionInput
from src.scrapbook import build_structured_memo, generate_followup_questions


CONTENT_TYPES = ["책", "소설", "논문", "기사", "영화", "뮤지컬", "연극", "시리즈", "기타"]
STATUSES = ["감상 완료", "감상 중", "중도하차", "재감상/이어보기"]


def render_memo(memo: dict[str, object]) -> None:
    st.subheader(memo.get("one_line_summary", "구조화된 감상 메모"))

    sections = [
        ("줄거리/핵심 내용", "content_summary"),
        ("인상 깊은 포인트", "impressive_points"),
        ("나의 해석", "personal_interpretation"),
        ("감정 기록", "emotional_trace"),
        ("다시 생각할 지점", "revisit_points"),
        ("비교 포인트", "comparison_points"),
        ("이어보기용 복습 메모", "review_note"),
    ]

    for label, key in sections:
        value = memo.get(key)
        if not value:
            continue
        st.markdown(f"**{label}**")
        if isinstance(value, list):
            for item in value:
                st.markdown(f"- {item}")
        else:
            st.write(value)

    tags = memo.get("tags") or []
    if tags:
        st.markdown("**태그**")
        st.write(" ".join(f"`{tag}`" for tag in tags))


def ensure_state() -> None:
    defaults = {
        "draft_input": None,
        "questions": [],
        "answers": {},
        "memo": None,
        "last_saved_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_reflection_input() -> ReflectionInput:
    return ReflectionInput(
        content_type=st.session_state["content_type"],
        title=st.session_state["title"].strip(),
        creator=st.session_state["creator"].strip() or None,
        status=st.session_state["status"],
        initial_note=st.session_state["initial_note"].strip(),
    )


def render_new_record(config: AppConfig) -> None:
    provider = st.sidebar.selectbox(
        "LLM provider",
        ["mock", "openai", "ollama"],
        index=["mock", "openai", "ollama"].index(config.llm_provider),
        help="mock은 API 키 없이 동작합니다. OpenAI/Ollama는 환경변수 설정 후 사용합니다.",
    )
    try:
        llm = create_llm_client(provider=provider, config=config)
    except LLMClientError as exc:
        st.warning(str(exc))
        st.info("API 키 없이 확인하려면 LLM provider를 mock으로 바꾸세요.")
        return

    st.header("새 감상 기록")

    with st.form("reflection_form"):
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.selectbox("작품 종류", CONTENT_TYPES, key="content_type")
        with cols[1]:
            st.selectbox("감상 상태", STATUSES, key="status")
        with cols[2]:
            st.text_input("작가/감독/창작자", key="creator", placeholder="선택 입력")

        st.text_input("작품 제목", key="title", placeholder="예: 프로젝트 헤일메리")
        st.text_area(
            "처음 떠오른 감상",
            key="initial_note",
            height=160,
            placeholder="짧은 문장, 키워드, 장면 묘사만 적어도 됩니다.",
        )
        submitted = st.form_submit_button("후속 질문 생성")

    if submitted:
        try:
            reflection_input = build_reflection_input()
            questions = generate_followup_questions(llm, reflection_input)
        except ValueError as exc:
            st.error(str(exc))
            return
        except LLMClientError as exc:
            st.error(str(exc))
            return

        st.session_state["draft_input"] = reflection_input.model_dump()
        st.session_state["questions"] = questions
        st.session_state["answers"] = {}
        st.session_state["memo"] = None
        st.session_state["last_saved_id"] = None

    if st.session_state["questions"]:
        st.divider()
        st.header("AI 후속 질문")
        st.caption("답하기 어려운 질문은 비워둬도 됩니다.")

        for idx, question in enumerate(st.session_state["questions"], start=1):
            st.text_area(
                f"Q{idx}. {question}",
                key=f"answer_{idx}",
                height=90,
            )

        if st.button("구조화 메모 만들기", type="primary"):
            try:
                reflection_input = ReflectionInput(**st.session_state["draft_input"])
                answers = {
                    question: st.session_state.get(f"answer_{idx}", "").strip()
                    for idx, question in enumerate(st.session_state["questions"], start=1)
                }
                memo = build_structured_memo(llm, reflection_input, answers)
            except LLMClientError as exc:
                st.error(str(exc))
                return

            st.session_state["answers"] = answers
            st.session_state["memo"] = memo

    if st.session_state["memo"]:
        st.divider()
        st.header("생성된 감상 스크랩")
        render_memo(st.session_state["memo"])

        if st.button("스크랩북에 저장"):
            reflection_input = ReflectionInput(**st.session_state["draft_input"])
            record_id = save_record(
                reflection_input=reflection_input,
                questions=st.session_state["questions"],
                answers=st.session_state["answers"],
                memo=st.session_state["memo"],
                provider=provider,
            )
            st.session_state["last_saved_id"] = record_id
            st.success(f"저장 완료: #{record_id}")


def render_archive() -> None:
    st.header("감상 아카이브")
    records = list_records()

    if not records:
        st.info("아직 저장된 감상 기록이 없습니다.")
        return

    options = {
        f"#{record['id']} {record['title']} - {record['content_type']} / {record['status']}": record["id"]
        for record in records
    }
    selected = st.selectbox("기록 선택", list(options.keys()))
    record = get_record(options[selected])
    if not record:
        st.error("기록을 찾을 수 없습니다.")
        return

    created_at = datetime.fromisoformat(record["created_at"]).strftime("%Y-%m-%d %H:%M")
    st.caption(f"{record['content_type']} · {record['status']} · {created_at}")
    if record.get("creator"):
        st.write(f"창작자: {record['creator']}")

    st.markdown("**처음 감상**")
    st.write(record["initial_note"])

    with st.expander("질문과 답변 보기"):
        answers = record["answers"]
        for question in record["questions"]:
            st.markdown(f"**Q. {question}**")
            st.write(answers.get(question) or "-")

    render_memo(record["memo"])


def main() -> None:
    config = AppConfig.from_env()
    st.set_page_config(page_title=config.app_name, layout="wide")
    ensure_state()
    init_db()

    st.title(config.app_name)
    st.caption("짧은 감상을 질문으로 확장해 다시 꺼내볼 수 있는 기록으로 바꿉니다.")

    tab_new, tab_archive = st.tabs(["새 기록", "아카이브"])
    with tab_new:
        render_new_record(config)
    with tab_archive:
        render_archive()


if __name__ == "__main__":
    main()
