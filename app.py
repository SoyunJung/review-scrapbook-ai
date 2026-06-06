from __future__ import annotations

import base64
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import AppConfig
from src.db import (
    add_record_asset,
    delete_record,
    get_record_embedding,
    get_record,
    get_similarity_reason,
    get_work,
    init_db,
    find_matching_works,
    list_records,
    list_record_assets,
    list_record_embeddings,
    list_record_versions,
    list_works,
    save_record,
    update_record,
    upsert_record_embedding,
    upsert_similarity_reason,
)
from src.llm import BaseLLMClient, LLMClientError, create_llm_client
from src.local_ai import (
    LocalAIDependencyError,
    LocalAIError,
    embed_texts,
    generate_question_audio,
    rank_embedding_matches,
    record_embedding_text,
    save_uploaded_audio,
    text_hash,
    transcribe_audio_file,
    tts_output_path,
    wav_audio_stats_from_bytes,
)
from src.models import ReflectionInput
from src.questions import DEFAULT_QUESTIONS
from src.scrapbook import (
    build_structured_memo,
    fallback_followup_question,
    generate_followup_questions,
    should_suggest_followup,
)
from src.work_context import (
    WorkContextError,
    external_context_source_label,
    fetch_wikipedia_context,
    format_external_context_for_prompt,
    has_external_context,
)


CONTENT_TYPES = ["책", "영화", "뮤지컬", "논문", "기사", "연극", "소설", "시리즈", "기타"]
STATUSES = ["감상 완료", "감상 중", "중도하차", "재감상/이어보기", "미지정"]
CONTENT_FILTERS = ["전체", "책", "영화", "뮤지컬", "연극", "논문", "기사", "소설", "시리즈", "기타"]
STATUS_FILTERS = ["전체", "감상 완료", "감상 중", "중도하차", "재감상/이어보기", "미지정"]
APPRECIATION_STAGES = ["초반부", "중반부", "완독 후", "재감상", "이어보기", "기타"]
INPUT_MODE_LABELS = {"text": "텍스트", "voice": "음성"}
DEFAULT_SPOILER_SCOPE = "입력 내용에서 추리한 감상 범위까지만"
SORT_OPTIONS = {
    "최근 수정순": "recent_updated",
    "최근 생성순": "recent_created",
    "감상일순": "appreciation_date",
}
SEARCH_MODES = ["키워드", "의미", "통합"]
MAX_FOLLOWUPS = 3
AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "webm"]
MEMO_FIELDS = [
    ("한 줄 요약", "one_line_summary", "text"),
    ("핵심 내용 / 맥락 메모", "content_context", "text"),
    ("인상 깊은 포인트", "impressive_points", "list"),
    ("나의 감정과 해석", "feelings_and_interpretation", "text"),
    ("비교 / 연상 포인트", "comparison_points", "text"),
    ("다시 생각할 지점", "revisit_points", "text"),
    ("이어보기용 복습 메모", "review_note", "text"),
    ("원문 단상", "original_notes", "text"),
    ("태그", "tags", "list"),
]
DETAIL_MEMO_FIELDS = [
    ("이어보기용 복습 메모", "review_note", "text"),
    ("인상 깊은 포인트", "impressive_points", "list"),
    ("나의 감정과 해석", "feelings_and_interpretation", "text"),
    ("비교 / 연상 포인트", "comparison_points", "text"),
    ("다시 생각할 지점", "revisit_points", "text"),
    ("핵심 내용 / 맥락 메모", "content_context", "text"),
    ("원문 단상", "original_notes", "text"),
]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #f6f7f5;
            --surface: #ffffff;
            --surface-soft: #f9faf8;
            --ink: #20231f;
            --muted: #667066;
            --line: #dfe5dd;
            --line-strong: #c8d2c6;
            --accent: #315f4f;
            --accent-soft: #e6eee9;
            --secondary: #5f5a78;
            --danger: #a34d4d;
        }

        .stApp {
            background: var(--page);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"],
        [data-testid="stHeaderActionElements"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .block-container {
            max-width: 780px;
            padding: 2rem 1rem 6rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        h1 {
            font-size: 2rem;
            margin-bottom: 0.25rem;
        }

        h2 {
            font-size: 1.35rem;
            margin-top: 0.7rem;
        }

        h3 {
            font-size: 1.05rem;
        }

        .quiet-caption {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0.25rem 0 1rem;
        }

        .context-source {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            margin: 0.25rem 0 0.85rem;
        }

        .context-source a {
            color: var(--accent);
            text-decoration: none;
            font-weight: 650;
        }

        .drawer-hero,
        .question-card,
        .work-card,
        .record-card,
        .memo-section,
        .empty-state,
        .detail-head,
        .skeleton-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: none;
        }

        .drawer-hero {
            background: var(--surface);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .drawer-title {
            font-size: 1.5rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .drawer-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .filter-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 760;
            margin: 0.45rem 0 0.15rem;
        }

        .inline-filter-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 760;
            line-height: 2.2rem;
            white-space: nowrap;
        }

        .list-header {
            padding: 0;
            margin: 0;
        }

        .list-title {
            font-size: 1.25rem;
            font-weight: 780;
            color: var(--ink);
            line-height: 1.35;
        }

        .list-subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.15rem;
        }

        .toolbar-divider {
            border-top: 1px solid var(--line);
            margin: 0.65rem 0 0.35rem;
        }

        [class*="st-key-screen_nav_"] {
            margin: 0 0 1rem;
            padding: 0.65rem 0.7rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
        }

        [class*="st-key-screen_nav_"] .stButton > button {
            width: 2.35rem;
            min-width: 2.35rem;
            min-height: 2.35rem;
            border-radius: 8px;
            border-color: var(--line-strong);
            background: var(--surface-soft);
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            padding: 0;
        }

        [class*="st-key-screen_nav_"] .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        [class*="st-key-screen_nav_"] [class*="st-key-inline_info_"] .stButton > button {
            width: auto;
            min-width: 5.4rem;
            padding: 0 0.75rem;
            font-size: 0.84rem;
            font-weight: 720;
        }

        .screen-toolbar {
            min-height: 2.35rem;
            padding: 0;
            margin: 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .screen-toolbar-title {
            color: var(--ink);
            font-size: 0.92rem;
            font-weight: 780;
            line-height: 1.2;
        }

        .screen-toolbar-subtitle {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.25;
            margin-top: 0.12rem;
        }

        .record-card {
            background: var(--surface);
            padding: 0.95rem;
            margin: 0.65rem 0 0.35rem;
        }

        .work-card {
            background: var(--surface);
            padding: 1rem;
            margin: 0.75rem 0 0.4rem;
        }

        .timeline-item {
            border-left: 2px solid var(--line-strong);
            padding: 0.15rem 0 0.9rem 1rem;
            margin-left: 0.4rem;
        }

        .timeline-date {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .mode-card {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            min-height: 7rem;
        }

        .record-title {
            font-size: 1.08rem;
            font-weight: 730;
            margin-bottom: 0.3rem;
        }

        .record-meta {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.7rem;
        }

        .record-summary {
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 0.75rem;
        }

        .similar-reason {
            color: var(--ink);
            background: var(--surface-soft);
            border-left: 3px solid var(--line-strong);
            padding: 0.7rem 0.85rem;
            border-radius: 0.45rem;
            font-size: 0.9rem;
            line-height: 1.55;
            margin-top: 0.35rem;
        }

        .record-foot {
            color: var(--accent);
            font-size: 0.82rem;
        }

        .question-card {
            background: var(--surface);
            padding: 1.15rem;
            margin: 0.9rem 0;
        }

        .voice-listening {
            background: var(--accent-soft);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0 0.9rem;
        }

        .voice-listening-title {
            color: var(--accent);
            font-size: 0.86rem;
            font-weight: 760;
            margin-bottom: 0.45rem;
        }

        .voice-listening-question {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 720;
            line-height: 1.55;
            margin: 0.35rem 0 0.75rem;
        }

        .voice-listening audio {
            width: 100%;
            margin-top: 0.25rem;
        }

        .step-label {
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 720;
            margin-bottom: 0.75rem;
        }

        .question-text {
            font-size: 1.22rem;
            font-weight: 760;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .followup-kicker {
            color: var(--secondary);
            font-size: 0.9rem;
            font-weight: 730;
            margin-bottom: 0.45rem;
        }

        .voice-panel,
        [class*="st-key-voice_answer_panel_"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.75rem 0 0.9rem;
        }

        .voice-panel-title {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .voice-panel-body {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 0.85rem;
        }

        .voice-answer-done {
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            margin: 0.4rem 0 0.75rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .transcript-preview {
            color: var(--ink);
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin: 0.45rem 0 0.85rem;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .transcript-preview-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .recording-status {
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.55rem 0 0.75rem;
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .recording-status.ok {
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid var(--line-strong);
        }

        .recording-status.warn {
            color: var(--danger);
            background: #fbf1f0;
            border: 1px solid #e8c8c4;
        }

        .mic-dot {
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            display: inline-block;
            background: var(--accent);
            box-shadow: 0 0 0 4px var(--accent-soft);
            flex: 0 0 auto;
        }

        .question-audio,
        [class*="st-key-question_audio_"] {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin: -0.25rem 0 0.85rem;
        }

        .question-audio-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 720;
            margin-bottom: 0.35rem;
        }

        [data-testid="stAudioInput"] {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
        }

        .memo-section {
            background: var(--surface);
            padding: 0.9rem;
            margin: 0.65rem 0;
        }

        .memo-label {
            color: var(--accent);
            font-size: 0.86rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .memo-body {
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .detail-head {
            background: var(--surface);
            padding: 1rem;
            margin: 0.75rem 0 1rem;
        }

        .tag {
            display: inline-block;
            margin: 0.15rem 0.25rem 0.15rem 0;
            padding: 0.18rem 0.45rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--accent);
            background: var(--accent-soft);
            font-size: 0.78rem;
        }

        .empty-state {
            background: var(--surface);
            padding: 1.2rem;
            color: var(--muted);
            line-height: 1.55;
        }

        .skeleton-card {
            background: var(--surface);
            padding: 1rem;
            margin: 1rem 0;
        }

        .skeleton-line {
            height: 0.85rem;
            margin: 0.8rem 0;
            border-radius: 8px;
            background: linear-gradient(90deg, #e4e9e2, #f4f6f3, #e4e9e2);
            background-size: 200% 100%;
            animation: pulse 1.2s ease-in-out infinite;
        }

        .skeleton-line.short { width: 45%; }
        .skeleton-line.mid { width: 72%; }

        @keyframes pulse {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--line);
            min-height: 2.75rem;
            font-weight: 700;
            background: var(--surface);
            color: var(--ink);
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }

        div.stButton > button[kind="primary"]:hover {
            background: #254b3f;
            border-color: #254b3f;
        }

        textarea,
        input,
        [data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: var(--line) !important;
            background-color: var(--surface) !important;
        }

        textarea:focus,
        input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        [data-testid="stSidebar"] {
            background: #eef2ee;
            border-right: 1px solid var(--line);
        }

        hr {
            border-color: var(--line);
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem;
        }

        .stProgress > div > div > div > div {
            background-color: var(--accent);
        }

        div[role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
        }

        [data-testid="stPills"] button {
            border-radius: 999px !important;
            min-height: 2.15rem;
            padding: 0.35rem 0.75rem;
        }

        [data-testid="stPills"] button[aria-pressed="true"] {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_memo(memo: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(memo)
    normalized.setdefault("content_context", normalized.get("content_summary", ""))
    normalized.setdefault(
        "feelings_and_interpretation",
        normalized.get("personal_interpretation")
        or normalized.get("emotional_trace")
        or "아직 기록되지 않았습니다.",
    )
    normalized.setdefault("comparison_points", "아직 기록되지 않았습니다.")
    normalized.setdefault("revisit_points", "아직 기록되지 않았습니다.")
    normalized.setdefault("review_note", normalized.get("content_context", ""))
    normalized.setdefault("original_notes", "")
    normalized.setdefault("tags", [])
    return normalized


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _to_list(value: str) -> list[str]:
    return [
        line.strip().lstrip("-").strip()
        for line in value.splitlines()
        if line.strip().lstrip("-").strip()
    ]


def _html_lines(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "아직 기록되지 않았습니다."
        return "<br>".join(f"- {escape(str(item))}" for item in value)
    return escape(str(value or "아직 기록되지 않았습니다."))


def _plain_memo_text(memo: dict[str, Any]) -> list[str]:
    normalized = normalize_memo(memo)
    values: list[str] = []
    for _, key, _ in MEMO_FIELDS:
        raw_value = normalized.get(key)
        if isinstance(raw_value, list):
            values.extend(str(item) for item in raw_value)
        elif raw_value:
            values.append(str(raw_value))
    return values


def make_search_snippet(record: dict[str, Any], memo: dict[str, Any], query: str) -> str:
    fallback = str(memo.get("one_line_summary") or "정리된 한 줄 요약이 없습니다.")
    query = query.strip()
    if not query:
        return fallback

    candidates = [
        str(record.get("title") or ""),
        str(record.get("work_context") or ""),
        str(record.get("spoiler_scope") or ""),
        str(record.get("free_text") or ""),
        str(record.get("initial_note") or ""),
        str(record.get("audio_transcript") or ""),
        *_plain_memo_text(memo),
        *(str(value) for value in (record.get("answers") or {}).values()),
    ]
    lowered_query = query.lower()
    for candidate in candidates:
        lowered = candidate.lower()
        index = lowered.find(lowered_query)
        if index < 0:
            continue
        start = max(0, index - 28)
        end = min(len(candidate), index + len(query) + 48)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(candidate) else ""
        return f"{prefix}{candidate[start:end]}{suffix}"
    return fallback


def _short_text(value: Any, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def similarity_prompt_payload(record: dict[str, Any]) -> dict[str, object]:
    memo = normalize_memo(record.get("edited_memo") or record.get("memo") or {})
    tags = memo.get("tags") if isinstance(memo.get("tags"), list) else []
    points = memo.get("impressive_points") if isinstance(memo.get("impressive_points"), list) else []
    return {
        "title": record.get("title") or "",
        "content_type": record.get("content_type") or "",
        "status": record.get("status") or "",
        "one_line_summary": _short_text(memo.get("one_line_summary"), limit=180),
        "feelings_and_interpretation": _short_text(memo.get("feelings_and_interpretation"), limit=280),
        "comparison_points": _short_text(memo.get("comparison_points"), limit=180),
        "review_note": _short_text(memo.get("review_note"), limit=180),
        "impressive_points": [_short_text(point, limit=90) for point in points[:4]],
        "tags": [str(tag) for tag in tags[:12]],
    }


def fallback_similarity_reason(source: dict[str, Any], target: dict[str, Any]) -> str:
    source_payload = similarity_prompt_payload(source)
    target_payload = similarity_prompt_payload(target)
    source_tags = {str(tag) for tag in source_payload.get("tags", []) if str(tag).strip()}
    target_tags = [str(tag) for tag in target_payload.get("tags", []) if str(tag).strip()]
    shared_tags = [tag for tag in target_tags if tag in source_tags][:3]
    target_summary = str(target_payload.get("one_line_summary") or target.get("title") or "").strip()
    if shared_tags:
        return (
            f"두 기록은 {', '.join(shared_tags)} 같은 감상 단서를 공유해요. "
            f"{target.get('title')}에서는 이 단서가 “{_short_text(target_summary, limit=90)}” 쪽으로 이어집니다."
        )
    return (
        f"{target.get('title')}도 작품 설명보다 감정의 방향과 다시 떠올릴 기억 단서가 중심인 기록이에요. "
        "현재 감상과 함께 보면 사용자가 어떤 장면과 감정을 오래 붙잡는지 비교해볼 수 있습니다."
    )


def similarity_reason_for_record_pair(
    source: dict[str, Any],
    target: dict[str, Any],
    llm: BaseLLMClient | None,
) -> str:
    source_id = int(source.get("id") or 0)
    target_id = int(target.get("id") or 0)
    source_embedding = get_record_embedding(source_id) if source_id else None
    target_embedding = get_record_embedding(target_id) if target_id else None
    source_hash = str((source_embedding or {}).get("source_hash") or "")
    target_hash = str((target_embedding or {}).get("source_hash") or "")
    provider_key = (
        f"{type(llm).__name__}:{getattr(llm, 'model', '')}"
        if llm is not None
        else "fallback"
    )

    if llm is not None and source_id and target_id and source_hash and target_hash:
        stored = get_similarity_reason(
            source_record_id=source_id,
            target_record_id=target_id,
            source_hash=source_hash,
            target_hash=target_hash,
            provider=provider_key,
        )
        if stored:
            return stored

    cache = st.session_state.setdefault("similarity_reason_cache", {})
    cache_key = ":".join(
        [
            "llm-reason-v1",
            str(source_id),
            source_hash or str(source.get("updated_at") or source.get("created_at") or ""),
            str(target_id),
            target_hash or str(target.get("updated_at") or target.get("created_at") or ""),
            provider_key,
        ]
    )
    if cache_key in cache:
        return str(cache[cache_key])

    fallback = fallback_similarity_reason(source, target)
    reason = fallback
    if llm is not None:
        try:
            generated = llm.explain_similarity(
                similarity_prompt_payload(source),
                similarity_prompt_payload(target),
            )
        except LLMClientError:
            generated = ""
        if generated.strip():
            reason = generated.strip()
            if source_id and target_id and source_hash and target_hash:
                upsert_similarity_reason(
                    source_record_id=source_id,
                    target_record_id=target_id,
                    source_hash=source_hash,
                    target_hash=target_hash,
                    provider=provider_key,
                    reason=reason,
                )
    cache[cache_key] = reason
    return reason


def build_record_text_for_embedding(record: dict[str, Any]) -> tuple[str, str]:
    memo = normalize_memo(record.get("edited_memo") or record.get("memo") or {})
    text = record_embedding_text(record, memo)
    return text, text_hash(text)


def index_record_embedding(record_id: int, config: AppConfig) -> str:
    record = get_record(record_id)
    if not record:
        raise LocalAIError("색인할 기록을 찾을 수 없습니다.")
    text, source_hash = build_record_text_for_embedding(record)
    model, vectors = embed_texts([text], config)
    if not vectors:
        raise LocalAIError("임베딩 결과가 비어 있습니다.")
    upsert_record_embedding(
        record_id=record_id,
        model=model,
        embedding=vectors[0],
        source_hash=source_hash,
    )
    return model


def rebuild_embedding_index(config: AppConfig) -> tuple[int, str]:
    records = list_records(sort_by="recent_updated")
    detailed_records = [get_record(record["id"]) for record in records]
    detailed_records = [record for record in detailed_records if record]
    texts_and_hashes = [build_record_text_for_embedding(record) for record in detailed_records]
    texts = [item[0] for item in texts_and_hashes]
    model, vectors = embed_texts(texts, config)
    for record, vector, (_, source_hash) in zip(detailed_records, vectors, texts_and_hashes):
        upsert_record_embedding(
            record_id=record["id"],
            model=model,
            embedding=vector,
            source_hash=source_hash,
        )
    return len(vectors), model


def stale_embedding_records(config: AppConfig) -> list[tuple[dict[str, Any], str, str]]:
    allowed_models = {config.embedding_model, config.embedding_fallback_model}
    stale: list[tuple[dict[str, Any], str, str]] = []
    for record_summary in list_records(sort_by="recent_updated"):
        record = get_record(record_summary["id"])
        if not record:
            continue
        text, source_hash = build_record_text_for_embedding(record)
        current = get_record_embedding(record["id"])
        if (
            not current
            or current.get("source_hash") != source_hash
            or str(current.get("model") or "") not in allowed_models
        ):
            stale.append((record, text, source_hash))
    return stale


def ensure_embedding_index_current(
    config: AppConfig,
    *,
    limit: int | None = None,
) -> tuple[int, str]:
    stale = stale_embedding_records(config)
    if limit is not None:
        stale = stale[:limit]
    if not stale:
        return 0, ""

    model, vectors = embed_texts([text for _, text, _ in stale], config)
    for (record, _text, source_hash), vector in zip(stale, vectors):
        upsert_record_embedding(
            record_id=record["id"],
            model=model,
            embedding=vector,
            source_hash=source_hash,
        )
    return len(vectors), model


def semantic_record_results(
    query: str,
    *,
    config: AppConfig,
    content_type: list[str],
    status: list[str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_embedding_index_current(config)
    model, vectors = embed_texts([query], config)
    del model
    candidates = list_record_embeddings()
    ranked = rank_embedding_matches(vectors[0], candidates, limit=limit * 3)
    allowed_records = {
        record["id"]: record
        for record in list_records(
            "",
            content_type=content_type,
            status=status,
            sort_by="recent_updated",
        )
    }

    results: list[dict[str, Any]] = []
    for item in ranked:
        record = allowed_records.get(item["record_id"])
        if not record:
            continue
        enriched = dict(record)
        enriched["_semantic_score"] = item["score"]
        results.append(enriched)
        if len(results) >= limit:
            break
    return results


def similar_records_for_detail(record: dict[str, Any], config: AppConfig) -> list[dict[str, Any]]:
    _text, source_hash = build_record_text_for_embedding(record)
    allowed_models = {config.embedding_model, config.embedding_fallback_model}
    current_embedding = get_record_embedding(record["id"])
    if (
        not current_embedding
        or current_embedding.get("source_hash") != source_hash
        or str(current_embedding.get("model") or "") not in allowed_models
    ):
        return []

    candidates = [
        item
        for item in list_record_embeddings()
        if item.get("record_id") != record["id"]
        and str(item.get("model") or "") in allowed_models
    ]
    ranked = rank_embedding_matches(current_embedding["embedding"], candidates, limit=5)
    results: list[dict[str, Any]] = []
    for item in ranked:
        detail = get_record(item["record_id"])
        if not detail:
            continue
        if detail.get("work_id") and detail.get("work_id") == record.get("work_id"):
            continue
        summary = {
            "id": detail["id"],
            "work_id": detail.get("work_id"),
            "title": detail["title"],
            "content_type": detail.get("content_type"),
            "status": detail.get("status"),
            "_semantic_score": item["score"],
        }
        results.append(summary)
    return results


def work_summary(work: dict[str, Any]) -> str:
    record_id = work.get("latest_record_id")
    if not record_id:
        return "아직 저장된 감상 요약이 없습니다."
    record = get_record(int(record_id))
    if not record:
        return "아직 저장된 감상 요약이 없습니다."
    memo = normalize_memo(record.get("edited_memo") or record.get("memo") or {})
    return str(memo.get("one_line_summary") or memo.get("review_note") or record.get("initial_note") or "요약 없음")


def merge_works(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for work in [*primary, *secondary]:
        work_id = int(work["id"])
        if work_id in seen:
            continue
        seen.add(work_id)
        merged.append(work)
    return merged


def works_from_semantic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    works: list[dict[str, Any]] = []
    best_scores: dict[int, float] = {}
    for record in records:
        work_id = record.get("work_id")
        if not work_id:
            continue
        score = float(record.get("_semantic_score") or 0)
        if score <= best_scores.get(int(work_id), -1):
            continue
        work = get_work(int(work_id))
        if not work:
            continue
        work["_semantic_score"] = score
        best_scores[int(work_id)] = score
        works = [item for item in works if int(item["id"]) != int(work_id)]
        works.append(work)
    works.sort(key=lambda item: item.get("_semantic_score", 0), reverse=True)
    return works


def merge_records(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in [*primary, *secondary]:
        record_id = int(record["id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        merged.append(record)
    return merged


def persist_draft_assets(record_id: int) -> None:
    audio_path = st.session_state.get("draft_audio_path")
    if audio_path:
        add_record_asset(
            record_id=record_id,
            asset_type="input_audio",
            path=str(audio_path),
            metadata={"source": "uploaded_audio"},
        )

    for question, path in st.session_state.get("draft_tts_assets", {}).items():
        add_record_asset(
            record_id=record_id,
            asset_type="tts_question",
            path=str(path),
            metadata={"question": question},
        )


def existing_question_audio(record_id: int, question: str) -> str | None:
    for asset in list_record_assets(record_id, asset_type="tts_question"):
        if (asset.get("metadata") or {}).get("question") == question:
            path = asset.get("path")
            if path and Path(path).exists():
                return str(path)
    return None


def transcribe_uploaded_audio(uploaded: Any, config: AppConfig):
    file_name = str(getattr(uploaded, "name", "") or "voice-answer.wav")
    audio_path = save_uploaded_audio(file_name, uploaded.getvalue())
    return transcribe_audio_file(audio_path, config)


def render_recording_status(uploaded: Any) -> None:
    stats = wav_audio_stats_from_bytes(uploaded.getvalue())
    if stats is None:
        return

    if stats.peak <= 8:
        st.markdown(
            f"""
            <div class="recording-status warn">
              녹음 파일은 만들어졌지만 목소리 신호가 거의 없어요.
              길이 {stats.duration_seconds:.1f}초 · 입력 음량 0에 가까움
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="recording-status ok">
          마이크 입력이 감지됐어요.
          길이 {stats.duration_seconds:.1f}초 · 입력 음량 확인됨
        </div>
        """,
        unsafe_allow_html=True,
    )


def remember_transcription_answer(
    *,
    target: dict[int, str],
    idx: int,
    text: str,
    audio_path: str,
    segments: list[dict[str, Any]],
) -> None:
    target[idx] = text.strip()
    st.session_state["draft_audio_path"] = audio_path
    st.session_state["draft_audio_transcript"] = "\n".join(
        part
        for part in [
            st.session_state.get("draft_audio_transcript", "").strip(),
            text.strip(),
        ]
        if part
    )
    st.session_state["draft_audio_segments"] = [
        *st.session_state.get("draft_audio_segments", []),
        *segments,
    ]


def question_audio_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".ogg", ".oga"}:
        return "audio/ogg"
    if suffix == ".webm":
        return "audio/webm"
    return "audio/wav"


def audio_data_uri(path: str | Path) -> str:
    audio_path = Path(path)
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{question_audio_mime_type(audio_path)};base64,{encoded}"


def should_autoplay_voice_question(key: str, question: str) -> bool:
    played = st.session_state.setdefault("draft_voice_autoplayed", {})
    question_key = text_hash(f"{key}|{question}")[:16]
    if played.get(key) == question_key:
        return False
    played[key] = question_key
    return True


def ensure_question_audio_path(
    question: str,
    *,
    config: AppConfig,
    record_id: int | None = None,
) -> str | None:
    session_assets = st.session_state.setdefault("draft_tts_assets", {})
    existing_path = (
        existing_question_audio(record_id, question)
        if record_id is not None
        else session_assets.get(question)
    )
    if existing_path and Path(existing_path).exists():
        return str(existing_path)

    with st.spinner("AI가 질문을 들려드릴 준비를 하고 있어요..."):
        output = tts_output_path(question, config, record_id=record_id)
        try:
            audio_path = generate_question_audio(question, output, config)
        except LocalAIDependencyError as exc:
            st.warning(str(exc))
            return None
        except LocalAIError as exc:
            st.error(str(exc))
            return None

    if record_id is not None:
        add_record_asset(
            record_id=record_id,
            asset_type="tts_question",
            path=str(audio_path),
            metadata={"question": question},
        )
    else:
        session_assets[question] = str(audio_path)
    return str(audio_path)


def render_auto_question_audio(
    question: str,
    *,
    config: AppConfig,
    key: str,
    record_id: int | None = None,
    kicker: str = "",
) -> None:
    audio_path = ensure_question_audio_path(question, config=config, record_id=record_id)
    audio_html = ""
    if audio_path:
        try:
            uri = audio_data_uri(audio_path)
            autoplay = " autoplay" if should_autoplay_voice_question(key, question) else ""
            audio_html = f'<audio controls{autoplay} src="{uri}"></audio>'
        except OSError:
            audio_html = ""
    kicker_html = (
        f'<div class="followup-kicker">{escape(kicker)}</div>'
        if kicker
        else ""
    )

    st.markdown(
        f"""
        <div class="voice-listening">
          <div class="voice-listening-title">AI가 질문을 들려드리고 있어요</div>
          {kicker_html}
          <div class="voice-listening-question">❝ {escape(question)} ❞</div>
          {audio_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_voice_question(
    question: str,
    *,
    config: AppConfig,
    key: str,
    label: str = "AI",
    kicker: str = "",
) -> None:
    del label
    render_auto_question_audio(question, config=config, key=key, kicker=kicker)


def render_question_audio_controls(
    question: str,
    *,
    config: AppConfig,
    key: str,
    record_id: int | None = None,
    button_label: str = "질문 다시 듣기",
) -> None:
    session_assets = st.session_state.setdefault("draft_tts_assets", {})
    existing_path = (
        existing_question_audio(record_id, question)
        if record_id is not None
        else session_assets.get(question)
    )
    with st.container(key=f"question_audio_{key}"):
        st.markdown('<div class="question-audio-label">질문 음성</div>', unsafe_allow_html=True)
        if existing_path and Path(existing_path).exists():
            st.audio(existing_path)

        if st.button(button_label, key=key, use_container_width=True):
            with st.spinner("질문 음성을 준비하고 있어요..."):
                output = tts_output_path(question, config, record_id=record_id)
                try:
                    audio_path = generate_question_audio(question, output, config)
                except LocalAIDependencyError as exc:
                    st.warning(str(exc))
                    return
                except LocalAIError as exc:
                    st.error(str(exc))
                    return

                if record_id is not None:
                    add_record_asset(
                        record_id=record_id,
                        asset_type="tts_question",
                        path=str(audio_path),
                        metadata={"question": question},
                    )
                else:
                    session_assets[question] = str(audio_path)
            st.rerun()


def render_audio_input_section(config: AppConfig, *, key_prefix: str) -> None:
    input_mode = st.session_state.get("draft_input_mode", "text")
    st.markdown("**음성 단상**")

    if input_mode == "voice" and hasattr(st, "audio_input"):
        with st.container(key=f"voice_answer_panel_{key_prefix}_free"):
            st.markdown(
                """
                <div class="voice-panel-title"><span class="mic-dot"></span>더 남기고 싶은 말이 있나요?</div>
                <div class="voice-panel-body">질문에 없었지만 기억하고 싶은 장면, 문장, 감정이 있다면 편하게 녹음해 주세요.</div>
                """,
                unsafe_allow_html=True,
            )
            recorded = st.audio_input("자유 단상 녹음", key=f"{key_prefix}_audio_record")
            if recorded is not None:
                render_recording_status(recorded)
            if recorded is not None and st.button("자유 단상 남기기", key=f"{key_prefix}_transcribe_recording", use_container_width=True):
                with st.spinner("음성 단상을 듣고 정리하고 있어요..."):
                    try:
                        result = transcribe_uploaded_audio(recorded, config)
                    except LocalAIDependencyError as exc:
                        st.warning(str(exc))
                        return
                    except LocalAIError as exc:
                        st.error(str(exc))
                        return
                st.session_state["draft_audio_path"] = result.audio_path
                st.session_state["draft_audio_transcript"] = "\n".join(
                    part
                    for part in [
                        st.session_state.get("draft_audio_transcript", "").strip(),
                        result.text.strip(),
                    ]
                    if part
                )
                st.session_state["draft_audio_segments"] = [
                    *st.session_state.get("draft_audio_segments", []),
                    *result.segments,
                ]
                st.success("추가 음성 단상을 기록했어요.")
                st.rerun()
    else:
        st.caption("선택 사항입니다. 음성 파일을 올린 뒤 전사 결과를 직접 수정할 수 있습니다.")
        uploaded = st.file_uploader(
            "음성 파일 업로드",
            type=AUDIO_TYPES,
            key=f"{key_prefix}_audio_upload",
        )
        if uploaded is not None and st.button("전사하기", key=f"{key_prefix}_transcribe", use_container_width=True):
            with st.spinner("음성 단상을 듣고 정리하고 있어요..."):
                try:
                    result = transcribe_uploaded_audio(uploaded, config)
                except LocalAIDependencyError as exc:
                    st.warning(str(exc))
                    return
                except LocalAIError as exc:
                    st.error(str(exc))
                    return
            st.session_state["draft_audio_path"] = result.audio_path
            st.session_state["draft_audio_transcript"] = result.text
            st.session_state["draft_audio_segments"] = result.segments
            st.success("전사 결과를 만들었습니다. 필요하면 아래에서 다듬어 주세요.")

    if str(st.session_state.get("draft_audio_transcript") or "").strip():
        with st.expander("전사 결과 확인 / 수정"):
            st.text_area(
                "전사 결과",
                key="draft_audio_transcript",
                height=120,
                placeholder="전사된 음성 단상이 여기에 표시됩니다.",
            )


def render_voice_answer_input(
    config: AppConfig,
    *,
    key_prefix: str,
    target: dict[int, str],
    idx: int,
) -> None:
    fallback_key = f"{key_prefix}_voice_text_fallback_{idx}"
    revision_key = f"{key_prefix}_voice_recording_revision_{idx}"
    st.session_state.setdefault(fallback_key, False)
    st.session_state.setdefault(revision_key, 0)

    with st.container(key=f"voice_answer_panel_{key_prefix}_{idx}"):
        saved_answer = str(target.get(idx, "") or "").strip()
        st.markdown(
            """
            <div class="voice-panel-title"><span class="mic-dot"></span>이제 편하게 말해주세요</div>
            <div class="voice-panel-body">정리된 문장으로 말하지 않아도 됩니다. 떠오르는 장면, 감정, 단어만 남겨도 나중에 감상 메모로 다듬어집니다.</div>
            """,
            unsafe_allow_html=True,
        )
        if saved_answer:
            st.markdown(
                f"""
                <div class="voice-answer-done">
                  음성 답변이 기록됐어요. 바로 다음으로 넘어가거나, 필요하면 다시 녹음할 수 있습니다.
                </div>
                <div class="transcript-preview">
                  <div class="transcript-preview-label">전사 내용</div>
                  {escape(saved_answer)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if hasattr(st, "audio_input"):
            recorded = st.audio_input(
                "답변 녹음",
                key=f"{key_prefix}_voice_recording_{idx}_{st.session_state[revision_key]}",
            )
        else:
            recorded = st.file_uploader(
                "답변 음성 파일",
                type=AUDIO_TYPES,
                key=f"{key_prefix}_voice_upload_{idx}_{st.session_state[revision_key]}",
                label_visibility="collapsed",
            )

        if recorded is not None:
            render_recording_status(recorded)

        cols = st.columns([1.15, 1, 1], gap="small")
        if cols[0].button(
            "답변 남기기",
            key=f"{key_prefix}_voice_transcribe_{idx}",
            type="primary",
            use_container_width=True,
            disabled=recorded is None,
        ):
            with st.spinner("음성 답변을 듣고 정리하고 있어요..."):
                try:
                    result = transcribe_uploaded_audio(recorded, config)
                except LocalAIDependencyError as exc:
                    st.warning(str(exc))
                    return
                except LocalAIError as exc:
                    st.error(str(exc))
                    return
            if not result.text.strip():
                st.warning("방금 답변을 잘 듣지 못했어요. 다시 말하거나 텍스트로 입력할 수 있어요.")
                return
            remember_transcription_answer(
                target=target,
                idx=idx,
                text=result.text,
                audio_path=result.audio_path,
                segments=result.segments,
            )
            st.session_state[fallback_key] = False
            st.success("음성 답변을 기록했어요.")
            st.rerun()

        if cols[1].button(
            "다시 녹음",
            key=f"{key_prefix}_voice_retry_{idx}",
            use_container_width=True,
            disabled=not saved_answer and recorded is None,
        ):
            target[idx] = ""
            st.session_state[revision_key] += 1
            st.session_state[fallback_key] = False
            st.rerun()

        if cols[2].button("텍스트 입력", key=f"{key_prefix}_voice_text_{idx}", use_container_width=True):
            st.session_state[fallback_key] = True
            st.rerun()

        if st.session_state.get(fallback_key):
            manual = st.text_area(
                "텍스트 답변",
                value=saved_answer,
                height=130,
                key=f"{key_prefix}_voice_manual_answer_{idx}",
                placeholder="녹음 대신 직접 적어도 됩니다.",
            )
            target[idx] = manual.strip()


def render_empty_state(
    *,
    title: str,
    body: str,
    button_label: str | None = None,
    button_key: str | None = None,
    on_click: Any | None = None,
) -> bool:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="record-title">{escape(title)}</div>
          <div class="drawer-subtitle">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if button_label:
        return st.button(
            button_label,
            key=button_key,
            type="primary",
            use_container_width=True,
            on_click=on_click,
        )
    return False


def render_memo(memo: dict[str, Any]) -> None:
    memo = normalize_memo(memo)
    st.subheader(str(memo.get("one_line_summary") or "감상 메모"))

    metadata = [
        str(memo.get("content_type") or "").strip(),
        str(memo.get("status") or "").strip(),
        str(memo.get("appreciation_date") or "").strip(),
    ]
    metadata = [item for item in metadata if item and item != "입력 없음"]
    if metadata:
        st.caption(" · ".join(metadata))

    for label, key, _ in DETAIL_MEMO_FIELDS:
        value = memo.get(key)
        if not value:
            continue
        st.markdown(
            f"""
            <div class="memo-section">
              <div class="memo-label">{escape(label)}</div>
              <div class="memo-body">{_html_lines(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tags = memo.get("tags") or []
    if tags:
        tag_html = " ".join(f'<span class="tag">{escape(str(tag))}</span>' for tag in tags)
        st.markdown(tag_html, unsafe_allow_html=True)


def render_record_versions(record_id: int) -> None:
    versions = list_record_versions(record_id)
    if not versions:
        return

    with st.expander(f"이전 버전 {len(versions)}개"):
        st.caption("이어 기록으로 AI가 글을 갱신하기 전 원본 스냅샷입니다. 메모 전체와 당시 입력/질문 답변을 그대로 보관합니다.")
        for version in versions:
            memo = normalize_memo(version.get("edited_memo") or version.get("memo") or {})
            saved_at = str(version.get("created_at") or "")[:16].replace("T", " ")
            record_updated = str(version.get("record_updated_at") or "")[:16].replace("T", " ")
            summary = str(memo.get("one_line_summary") or "요약 없음")
            st.markdown(
                f"""
                <div class="record-card">
                  <div class="record-title">{escape(str(version.get("version_label") or "이전 버전"))}</div>
                  <div class="record-meta">보관 {escape(saved_at)} · 이전 수정 {escape(record_updated)}</div>
                  <div class="record-summary">“{escape(summary)}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_memo(memo)
            with st.expander("이 버전의 원본 입력 / 질문 답변"):
                if str(version.get("initial_note") or "").strip():
                    st.markdown("**초기 단상**")
                    st.write(version["initial_note"])
                if str(version.get("free_text") or "").strip():
                    st.markdown("**자유 단상**")
                    st.write(version["free_text"])
                if str(version.get("audio_transcript") or "").strip():
                    st.markdown("**음성 전사**")
                    st.write(version["audio_transcript"])
                answers = version.get("answers") or {}
                if answers:
                    st.markdown("**질문과 답변**")
                    for question, answer in answers.items():
                        st.markdown(f"**Q. {question}**")
                        st.write(answer or "건너뜀")


def render_memo_editor(
    memo: dict[str, Any],
    *,
    key_prefix: str,
    title: str = "감상 메모 초안",
) -> None:
    memo = normalize_memo(memo)
    st.header(title)
    st.markdown('<p class="quiet-caption">내 기록처럼 다듬은 뒤 저장하세요.</p>', unsafe_allow_html=True)

    for label, key, field_type in MEMO_FIELDS:
        height = 90 if key in {"one_line_summary", "tags"} else 135
        value = _to_text(memo.get(key))
        help_text = "한 줄에 하나씩 입력" if field_type == "list" else None
        st.text_area(label, value=value, height=height, key=f"{key_prefix}_{key}", help=help_text)


def read_edited_memo(reflection_input: ReflectionInput, *, key_prefix: str) -> dict[str, Any]:
    edited: dict[str, Any] = {
        "title": reflection_input.title,
        "content_type": reflection_input.content_type,
        "appreciation_date": reflection_input.appreciation_date or "입력 없음",
        "status": reflection_input.status,
    }
    for _, key, field_type in MEMO_FIELDS:
        raw_value = str(st.session_state.get(f"{key_prefix}_{key}", "")).strip()
        edited[key] = _to_list(raw_value) if field_type == "list" else raw_value
    return edited


def set_draft_external_context(
    external_context: dict[str, Any] | None,
    config: AppConfig | None = None,
) -> None:
    max_chars = (
        config.wikipedia_prompt_max_chars
        if config is not None
        else AppConfig.wikipedia_prompt_max_chars
    )
    payload = external_context if has_external_context(external_context) else {}
    st.session_state["draft_external_context"] = payload
    st.session_state["draft_external_context_prompt"] = format_external_context_for_prompt(
        payload,
        max_chars=max_chars,
    )


def clear_draft_external_context() -> None:
    st.session_state["draft_external_context"] = {}
    st.session_state["draft_external_context_prompt"] = ""
    st.session_state["draft_external_context_notice"] = ""


def fetch_and_store_draft_external_context(
    *,
    title: str,
    content_type: str,
    creator: str,
    config: AppConfig,
) -> None:
    clear_draft_external_context()
    if not config.wikipedia_context_enabled:
        return
    try:
        external_context = fetch_wikipedia_context(
            title,
            config,
            content_type=content_type,
            creator=creator,
        )
    except WorkContextError as exc:
        st.session_state["draft_external_context_notice"] = str(exc)
        return

    if has_external_context(external_context):
        set_draft_external_context(external_context, config)
        st.session_state["draft_external_context_notice"] = (
            f"{external_context_source_label(external_context)} 작품 맥락을 참고합니다."
        )


def render_draft_external_context_status() -> None:
    context = st.session_state.get("draft_external_context") or {}
    notice = str(st.session_state.get("draft_external_context_notice") or "").strip()
    if has_external_context(context):
        label = external_context_source_label(context)
        url = str(context.get("page_url") or "").strip()
        link_html = (
            f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
            if url
            else escape(label)
        )
        st.markdown(
            f"""
            <div class="context-source">
              작품 맥락 보조 자료: {link_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    if notice:
        st.caption(notice)


def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "screen": "home",
        "selected_record_id": None,
        "selected_work_id": None,
        "draft_work_id": None,
        "draft_existing_record_id": None,
        "draft_input_mode": "text",
        "draft_appreciation_stage": APPRECIATION_STAGES[0],
        "draft_force_new_work": False,
        "draft_title": "",
        "draft_creator": "",
        "draft_content_type": CONTENT_TYPES[0],
        "draft_status": STATUSES[0],
        "draft_work_context": "",
        "draft_external_context": {},
        "draft_external_context_prompt": "",
        "draft_external_context_notice": "",
        "draft_spoiler_scope": DEFAULT_SPOILER_SCOPE,
        "draft_use_date": True,
        "draft_date": date.today(),
        "question_index": 0,
        "show_followup": False,
        "draft_base_answers": {},
        "draft_followups": {},
        "draft_followup_answers": {},
        "draft_free_text": "",
        "draft_audio_path": "",
        "draft_audio_transcript": "",
        "draft_audio_segments": [],
        "draft_tts_assets": {},
        "draft_voice_autoplayed": {},
        "memo": None,
        "last_saved_id": None,
        "detail_editing_record_id": None,
        "delete_confirm_record_id": None,
        "post_delete_notice": "",
        "home_search": "",
        "home_search_mode": SEARCH_MODES[0],
        "home_content_filters": [],
        "home_status_filters": [],
        "home_sort": next(iter(SORT_OPTIONS)),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_draft() -> None:
    for key in [
        "info_title",
        "info_creator",
        "info_content_type",
        "info_status",
        "info_use_date",
        "info_date",
        "draft_title",
        "draft_creator",
        "draft_content_type",
        "draft_status",
        "draft_work_context",
        "draft_external_context",
        "draft_external_context_prompt",
        "draft_external_context_notice",
        "draft_spoiler_scope",
        "draft_use_date",
        "draft_date",
        "question_index",
        "show_followup",
        "draft_base_answers",
        "draft_followups",
        "draft_followup_answers",
        "draft_free_text",
        "draft_audio_path",
        "draft_audio_transcript",
        "draft_audio_segments",
        "draft_tts_assets",
        "draft_voice_autoplayed",
        "draft_work_id",
        "draft_existing_record_id",
        "draft_input_mode",
        "draft_appreciation_stage",
        "draft_force_new_work",
        "memo",
        "last_saved_id",
    ]:
        st.session_state.pop(key, None)
    ensure_state()


def set_screen(screen: str) -> None:
    st.session_state["screen"] = screen


def exit_record_flow_to_home() -> None:
    reset_draft()
    set_screen("home")


def render_toolbar_label(title: str, subtitle: str = "") -> None:
    subtitle_html = (
        f'<div class="screen-toolbar-subtitle">{escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="screen-toolbar">
          <div class="screen-toolbar-title">{escape(title)}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_record_flow_nav(
    key_prefix: str,
    *,
    title: str,
    subtitle: str = "",
    show_info_back: bool = False,
) -> None:
    with st.container(
        key=f"screen_nav_{key_prefix}",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
    ):
        if st.button("<", key=f"{key_prefix}_home_back", help="홈으로"):
            exit_record_flow_to_home()
            st.rerun()
        if show_info_back:
            with st.container(key=f"inline_info_{key_prefix}"):
                if st.button("작품 정보", key=f"{key_prefix}_info_back"):
                    set_screen("info")
                    st.rerun()
        render_toolbar_label(title, subtitle)


def render_home_back_nav(key_prefix: str, *, title: str, subtitle: str = "") -> None:
    with st.container(
        key=f"screen_nav_{key_prefix}",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
    ):
        if st.button("<", key=f"{key_prefix}_home_back", help="홈으로"):
            set_screen("home")
            st.rerun()
        render_toolbar_label(title, subtitle)


def clear_home_filters() -> None:
    st.session_state["home_search"] = ""
    st.session_state["home_search_mode"] = SEARCH_MODES[0]
    st.session_state["home_content_filters"] = []
    st.session_state["home_status_filters"] = []
    st.session_state["home_sort"] = next(iter(SORT_OPTIONS))


def continuation_context_from_record(record: dict[str, Any]) -> str:
    memo = normalize_memo(record.get("edited_memo") or record.get("memo") or {})
    answers = record.get("answers") or {}
    answer_notes = [
        str(value).strip()
        for value in answers.values()
        if str(value or "").strip()
    ][:5]
    parts = [
        "이 작품은 이미 하나의 감상 글로 저장되어 있습니다.",
        "이번 입력은 새 글을 추가하는 것이 아니라 기존 감상 글을 업데이트하기 위한 이어 기록입니다.",
        "기존 글의 핵심은 보존하되, 새 답변에서 나온 감정과 장면을 반영해 더 밀도 있게 갱신하세요.",
        f"기존 한 줄 요약: {memo.get('one_line_summary') or record.get('initial_note') or '없음'}",
        f"기존 이어보기 메모: {memo.get('review_note') or '없음'}",
    ]
    if record.get("work_context"):
        parts.append(f"작품 맥락: {record['work_context']}")
    if answer_notes:
        parts.append("이전 답변 단서:\n- " + "\n- ".join(answer_notes))
    return "\n\n".join(parts)


def start_record_for_work(
    work: dict[str, Any],
    *,
    input_mode: str | None = None,
    external_context: dict[str, Any] | None = None,
) -> None:
    incoming_external_context = (
        external_context
        if has_external_context(external_context)
        else st.session_state.get("draft_external_context") or {}
    )
    reset_draft()
    st.session_state["draft_work_id"] = work["id"]
    st.session_state["draft_title"] = work["title"]
    st.session_state["draft_creator"] = work.get("creator") or ""
    st.session_state["draft_content_type"] = work.get("content_type") or CONTENT_TYPES[0]
    st.session_state["draft_status"] = work.get("status") or STATUSES[0]
    record_id = work.get("latest_record_id")
    record = get_record(int(record_id)) if record_id else None
    if record:
        st.session_state["draft_existing_record_id"] = record["id"]
        st.session_state["draft_work_context"] = continuation_context_from_record(record)
        st.session_state["draft_spoiler_scope"] = record.get("spoiler_scope") or DEFAULT_SPOILER_SCOPE
        set_draft_external_context(
            record.get("external_context") or incoming_external_context,
        )
    else:
        st.session_state.setdefault("draft_work_context", "")
        st.session_state.setdefault("draft_spoiler_scope", DEFAULT_SPOILER_SCOPE)
        set_draft_external_context(incoming_external_context)
    if input_mode:
        st.session_state["draft_input_mode"] = input_mode
        set_screen("question")
    else:
        set_screen("mode")


def restore_draft_work_fields() -> bool:
    if str(st.session_state.get("draft_title") or "").strip():
        return True

    work_id = st.session_state.get("draft_work_id")
    if not work_id:
        return False

    try:
        work = get_work(int(work_id))
    except (TypeError, ValueError):
        work = None
    if not work:
        return False

    st.session_state["draft_title"] = str(work.get("title") or "").strip()
    st.session_state["draft_creator"] = work.get("creator") or ""
    st.session_state["draft_content_type"] = work.get("content_type") or CONTENT_TYPES[0]
    st.session_state["draft_status"] = work.get("status") or STATUSES[0]
    return bool(str(st.session_state.get("draft_title") or "").strip())


def render_missing_draft_title() -> None:
    st.error("콘텐츠 제목 정보가 비어 있습니다. 작품 정보를 먼저 입력해주세요.")
    cols = st.columns([1, 1])
    if cols[0].button("홈으로", use_container_width=True):
        exit_record_flow_to_home()
        st.rerun()
    if cols[1].button("작품 정보 입력으로 돌아가기", use_container_width=True):
        reset_draft()
        set_screen("info")
        st.rerun()


def build_reflection_input_from_state() -> ReflectionInput:
    if not restore_draft_work_fields():
        raise ValueError("콘텐츠 제목 정보가 비어 있습니다. 작품 정보를 먼저 입력해주세요.")

    appreciation_date = None
    if st.session_state.get("draft_use_date"):
        raw_date = st.session_state.get("draft_date")
        if isinstance(raw_date, date):
            appreciation_date = raw_date.isoformat()

    base_answers = st.session_state.get("draft_base_answers", {})
    first_note = next((answer for _, answer in sorted(base_answers.items()) if answer), "")
    free_text = st.session_state.get("draft_free_text", "").strip()
    audio_transcript = st.session_state.get("draft_audio_transcript", "").strip()

    return ReflectionInput(
        content_type=st.session_state.get("draft_content_type", "미지정"),
        title=str(st.session_state.get("draft_title", "")).strip(),
        creator=st.session_state.get("draft_creator", "").strip() or None,
        appreciation_date=appreciation_date,
        status=st.session_state.get("draft_status", "미지정"),
        work_context=str(st.session_state.get("draft_work_context", "")).strip(),
        external_context=str(st.session_state.get("draft_external_context_prompt", "")).strip(),
        external_context_source=external_context_source_label(
            st.session_state.get("draft_external_context") or {}
        ),
        spoiler_scope=DEFAULT_SPOILER_SCOPE,
        initial_note=first_note or free_text or audio_transcript,
        free_text=free_text,
        audio_transcript=audio_transcript,
    )


def collect_all_answers_from_state() -> dict[str, str]:
    answers: dict[str, str] = {}
    base_answers = st.session_state.get("draft_base_answers", {})
    for idx, question in enumerate(DEFAULT_QUESTIONS):
        answers[question] = base_answers.get(idx, "")

    followups = st.session_state.get("draft_followups", {})
    followup_answers = st.session_state.get("draft_followup_answers", {})
    for idx, question in followups.items():
        answers[question] = followup_answers.get(idx, "")

    free_text = st.session_state.get("draft_free_text", "").strip()
    if free_text:
        answers["자유 텍스트 단상"] = free_text
    audio_transcript = st.session_state.get("draft_audio_transcript", "").strip()
    if audio_transcript:
        answers["음성 전사 단상"] = audio_transcript
    return answers


def validate_reflection_input(reflection_input: ReflectionInput, answers: dict[str, str]) -> None:
    if not reflection_input.title:
        raise ValueError("콘텐츠 제목은 필수입니다.")
    if (
        not any(answer.strip() for answer in answers.values())
        and not reflection_input.free_text
        and not reflection_input.audio_transcript
    ):
        raise ValueError("질문 중 하나 이상 또는 자유 단상을 입력하세요.")


def reflection_input_from_record(record: dict[str, Any]) -> ReflectionInput:
    return ReflectionInput(
        content_type=record.get("content_type") or "미지정",
        title=record["title"],
        creator=record.get("creator"),
        appreciation_date=record.get("appreciation_date"),
        status=record.get("status") or "미지정",
        work_context=record.get("work_context") or "",
        external_context=format_external_context_for_prompt(record.get("external_context") or {}),
        external_context_source=external_context_source_label(record.get("external_context") or {}),
        spoiler_scope=record.get("spoiler_scope") or DEFAULT_SPOILER_SCOPE,
        initial_note=record.get("initial_note") or "",
        free_text=record.get("free_text") or "",
        audio_transcript=record.get("audio_transcript") or "",
    )


def render_provider_selector(config: AppConfig):
    with st.sidebar:
        provider = st.selectbox(
            "LLM provider",
            ["mock", "openai", "ollama"],
            index=["mock", "openai", "ollama"].index(config.llm_provider),
        )
        model_label = {
            "mock": "로컬 흐름 확인용",
            "openai": config.openai_model,
            "ollama": config.ollama_model,
        }[provider]
        st.caption(f"현재 모델: {model_label}")
        if provider == "openai":
            st.caption("입력한 감상 텍스트가 OpenAI API로 전송됩니다.")
        elif provider == "ollama":
            st.caption("Ollama 서버가 실행 중이어야 합니다.")

        st.divider()
        st.markdown("**로컬 AI**")
        st.caption(f"STT: faster-whisper `{config.stt_model}`")
        st.caption(f"Embedding: `{config.embedding_model}`")
        st.caption(f"TTS: `{config.tts_model}` / {config.tts_speaker}")
        st.caption(
            f"작품 맥락: Wikipedia `{config.wikipedia_language}` "
            + ("자동 참고" if config.wikipedia_context_enabled else "꺼짐")
        )
        with st.expander("관리용 색인"):
            st.caption("저장/수정 때 색인을 갱신합니다. 연결된 감상은 저장된 색인만 빠르게 조회합니다.")
            if st.button("누락/변경분 보정", use_container_width=True):
                try:
                    count, model = ensure_embedding_index_current(config)
                except LocalAIDependencyError as exc:
                    st.warning(str(exc))
                except LocalAIError as exc:
                    st.error(str(exc))
                else:
                    if count:
                        st.success(f"{count}개 기록을 `{model}`로 보정했습니다.")
                    else:
                        st.success("이미 최신 색인입니다.")
            if st.button("전체 다시 만들기", use_container_width=True):
                try:
                    count, model = rebuild_embedding_index(config)
                except LocalAIDependencyError as exc:
                    st.warning(str(exc))
                except LocalAIError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"{count}개 기록을 `{model}`로 색인했습니다.")
    try:
        return provider, create_llm_client(provider=provider, config=config)
    except LLMClientError as exc:
        st.sidebar.warning(str(exc))
        return provider, None


def render_home(config: AppConfig) -> None:
    st.markdown(
        """
        <div class="drawer-hero">
          <div class="drawer-title">내 감상 서랍</div>
          <div class="drawer-subtitle">감상했던 순간을 다시 꺼내보세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_cols = st.columns([3.2, 1.3, 1.45], gap="small", vertical_alignment="bottom")
    with search_cols[0]:
        search_query = st.text_input(
            "검색",
            placeholder="작품명, 장면, 감정, 키워드로 검색",
            key="home_search",
            label_visibility="collapsed",
        )
    with search_cols[1]:
        search_mode = st.selectbox(
            "검색 방식",
            SEARCH_MODES,
            key="home_search_mode",
            label_visibility="collapsed",
        )
    with search_cols[2]:
        if st.button("+ 새 감상 기록", type="primary", use_container_width=True):
            reset_draft()
            set_screen("info")
            st.rerun()

    selected_content_types = st.session_state.get("home_content_filters", []) or []
    selected_statuses = st.session_state.get("home_status_filters", []) or []
    sort_label = st.session_state.get("home_sort", next(iter(SORT_OPTIONS)))

    has_active_filters = (
        bool(search_query.strip())
        or bool(selected_content_types)
        or bool(selected_statuses)
    )

    keyword_query = search_query if search_mode in {"키워드", "통합"} else ""
    records = list_records(
        keyword_query,
        content_type=selected_content_types,
        status=selected_statuses,
        sort_by=SORT_OPTIONS[sort_label],
    )
    if search_query.strip() and search_mode in {"의미", "통합"}:
        try:
            semantic_records = semantic_record_results(
                search_query,
                config=config,
                content_type=selected_content_types,
                status=selected_statuses,
            )
        except LocalAIDependencyError as exc:
            st.warning(str(exc))
            semantic_records = []
        except LocalAIError as exc:
            st.error(str(exc))
            semantic_records = []
        records = semantic_records if search_mode == "의미" else merge_records(semantic_records, records)
    header_cols = st.columns([4.6, 1.4], gap="medium", vertical_alignment="bottom")
    with header_cols[0]:
        st.markdown(
            f"""
            <div class="list-header">
              <div class="list-title">기록</div>
              <div class="list-subtitle">{len(records)}개의 기록</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown('<div class="filter-label">정렬</div>', unsafe_allow_html=True)
        st.selectbox(
            "정렬",
            list(SORT_OPTIONS),
            key="home_sort",
            label_visibility="collapsed",
        )

    with st.container(border=False):
        type_cols = st.columns([0.85, 5.15], gap="small", vertical_alignment="center")
        with type_cols[0]:
            st.markdown('<div class="inline-filter-label">유형</div>', unsafe_allow_html=True)
        with type_cols[1]:
            selected_content_types = st.pills(
                "콘텐츠 유형",
                CONTENT_FILTERS[1:],
                selection_mode="multi",
                key="home_content_filters",
                width="stretch",
                label_visibility="collapsed",
            )
            selected_content_types = selected_content_types or []

        status_cols = st.columns([0.85, 5.15], gap="small", vertical_alignment="center")
        with status_cols[0]:
            st.markdown('<div class="inline-filter-label">상태</div>', unsafe_allow_html=True)
        with status_cols[1]:
            selected_statuses = st.pills(
                "감상 상태",
                STATUS_FILTERS[1:],
                selection_mode="multi",
                key="home_status_filters",
                width="stretch",
                label_visibility="collapsed",
            )
            selected_statuses = selected_statuses or []

    st.markdown('<div class="toolbar-divider"></div>', unsafe_allow_html=True)

    if not records:
        if has_active_filters:
            render_empty_state(
                title="검색 결과가 없어요.",
                body="다른 작품명이나 감정 키워드로 검색해보세요. 예: 찝찝함, 마지막 장면, 외로움, 다시 읽기",
                button_label="전체 기록 보기",
                button_key="clear_home_filters",
                on_click=clear_home_filters,
            )
            return

        if render_empty_state(
            title="아직 남긴 감상이 없어요.",
            body="질문에 짧게 답하면 나중에 다시 읽기 좋은 감상 메모가 만들어져요.",
            button_label="첫 감상 기록하기",
            button_key="first_record",
        ):
            reset_draft()
            set_screen("info")
            st.rerun()
        return

    for record in records[:20]:
        detail = get_record(record["id"])
        memo = normalize_memo((detail or {}).get("edited_memo") or (detail or {}).get("memo") or {})
        summary_text = make_search_snippet(detail or record, memo, keyword_query)
        points = memo.get("impressive_points") or []
        point_count = len(points) if isinstance(points, list) else 0
        has_review_note = bool(str(memo.get("review_note") or "").strip())
        date_label = str(
            record.get("appreciation_date")
            or record.get("updated_at")
            or record.get("created_at", "")
        )[:10]
        meta = [
            str(record.get("content_type") or "미지정"),
            str(record.get("status") or "미지정"),
            date_label,
        ]
        foot_items = [f"인상 깊은 포인트 {point_count}개"]
        foot_items.append("이어보기 메모 있음" if has_review_note else "이어보기 메모 없음")
        if search_query.strip():
            foot_items.append("검색 맥락 표시")
        if record.get("_semantic_score") is not None:
            foot_items.append(f"유사도 {record['_semantic_score']:.2f}")
        foot = " · ".join(foot_items)
        st.markdown(
            f"""
            <div class="record-card">
              <div class="record-title">{escape(str(record["title"]))}</div>
              <div class="record-meta">{escape(" · ".join(meta))}</div>
              <div class="record-summary">“{escape(summary_text)}”</div>
              <div class="record-foot">{escape(foot)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("열기", key=f"open_{record['id']}", use_container_width=True):
            st.session_state["selected_record_id"] = record["id"]
            set_screen("detail")
            st.rerun()


def render_info_screen(config: AppConfig) -> None:
    render_record_flow_nav("info", title="새 감상 기록", subtitle="작품 정보")
    st.header("오늘 어떤 콘텐츠를 기록할까요?")
    st.markdown('<p class="quiet-caption">제목만 적어도 질문으로 바로 시작할 수 있습니다.</p>', unsafe_allow_html=True)

    st.session_state.setdefault("info_title", st.session_state.get("draft_title", ""))
    st.session_state.setdefault("info_creator", st.session_state.get("draft_creator", ""))
    st.session_state.setdefault("info_content_type", st.session_state.get("draft_content_type", CONTENT_TYPES[0]))
    st.session_state.setdefault("info_status", st.session_state.get("draft_status", STATUSES[0]))
    st.session_state.setdefault("info_use_date", st.session_state.get("draft_use_date", True))
    st.session_state.setdefault("info_date", st.session_state.get("draft_date", date.today()))

    with st.form("content_info"):
        st.text_input("콘텐츠 제목", key="info_title", placeholder="예: 위키드, 프로젝트 헤일메리, 데미안")
        st.text_input("작가/감독/창작자", key="info_creator", placeholder="선택 입력")
        st.radio("콘텐츠 유형", CONTENT_TYPES, horizontal=True, key="info_content_type")
        st.radio("감상 상태", STATUSES, horizontal=True, key="info_status")
        st.checkbox("감상일 남기기", key="info_use_date")
        if st.session_state.get("info_use_date", True):
            st.date_input("감상일", key="info_date")

        start = st.form_submit_button("질문으로 기록 시작하기", type="primary", use_container_width=True)

    if start:
        title = str(st.session_state.get("info_title", "")).strip()
        if not title:
            st.error("콘텐츠 제목을 입력하세요.")
            return
        st.session_state["draft_title"] = title
        st.session_state["draft_creator"] = str(st.session_state.get("info_creator", "")).strip()
        st.session_state["draft_content_type"] = st.session_state.get("info_content_type", CONTENT_TYPES[0])
        st.session_state["draft_status"] = st.session_state.get("info_status", STATUSES[0])
        st.session_state["draft_work_context"] = ""
        st.session_state["draft_spoiler_scope"] = DEFAULT_SPOILER_SCOPE
        st.session_state["draft_use_date"] = bool(st.session_state.get("info_use_date", True))
        st.session_state["draft_date"] = st.session_state.get("info_date", date.today())
        st.session_state["question_index"] = 0
        st.session_state["show_followup"] = False
        st.session_state["draft_base_answers"] = {}
        st.session_state["draft_followups"] = {}
        st.session_state["draft_followup_answers"] = {}
        st.session_state["memo"] = None
        st.session_state["draft_force_new_work"] = False
        st.session_state["draft_work_id"] = None
        with st.spinner("Wikipedia EN에서 작품 맥락을 확인하고 있어요."):
            fetch_and_store_draft_external_context(
                title=title,
                content_type=st.session_state.get("draft_content_type", ""),
                creator=st.session_state.get("draft_creator", ""),
                config=config,
            )
        matches = find_matching_works(
            title,
            st.session_state.get("draft_content_type"),
        )
        set_screen("work_match" if matches else "mode")
        st.rerun()


def maybe_offer_followup(llm: BaseLLMClient, idx: int, answer: str) -> bool:
    if not answer.strip():
        return False
    followups = st.session_state.setdefault("draft_followups", {})
    if idx in followups or len(followups) >= MAX_FOLLOWUPS:
        return False

    try:
        reflection_input = build_reflection_input_from_state()
        answers = collect_all_answers_from_state()
        candidates = generate_followup_questions(llm, reflection_input, answers)
    except ValueError as exc:
        st.warning(str(exc))
        return False
    except LLMClientError:
        candidates = []

    existing = set(followups.values())
    for candidate in candidates:
        if candidate not in existing and candidate not in DEFAULT_QUESTIONS:
            followups[idx] = candidate
            return True
    if should_suggest_followup(answer):
        fallback = fallback_followup_question(idx)
        if fallback not in existing and fallback not in DEFAULT_QUESTIONS:
            followups[idx] = fallback
            return True
    return False


def advance_question() -> None:
    idx = st.session_state["question_index"]
    if idx + 1 >= len(DEFAULT_QUESTIONS):
        set_screen("free")
    else:
        st.session_state["question_index"] = idx + 1
        st.session_state["show_followup"] = False


def render_question_screen(llm: BaseLLMClient | None, config: AppConfig) -> None:
    if llm is None:
        st.warning("LLM provider 설정을 확인하세요.")
        return

    idx = st.session_state["question_index"]
    if not restore_draft_work_fields():
        render_missing_draft_title()
        return

    total = len(DEFAULT_QUESTIONS)
    render_record_flow_nav("question", title="감상 인터뷰", subtitle=f"{idx + 1} / {total}")
    progress = (idx + 1) / total
    st.progress(progress)
    input_mode = st.session_state.get("draft_input_mode", "text")
    if input_mode == "voice":
        st.caption("AI가 질문을 건네고, 답변은 마이크로 짧게 남기는 흐름입니다. 전사 원문은 저장 후 상세에서 확인할 수 있어요.")

    if st.session_state.get("show_followup"):
        question = st.session_state["draft_followups"].get(idx)
        if input_mode == "voice":
            render_voice_question(
                question or "",
                config=config,
                key=f"draft_followup_tts_{idx}",
                kicker="조금만 더 남겨볼까요?",
            )
        else:
            st.markdown(
                f"""
                <div class="question-card">
                  <div class="followup-kicker">조금만 더 남겨볼까요?</div>
                  <div class="question-text">{escape(question or "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        answer_key = f"followup_input_{idx}"
        current_value = st.session_state["draft_followup_answers"].get(idx, "")
        if input_mode == "voice":
            render_voice_answer_input(
                config,
                key_prefix="followup",
                target=st.session_state["draft_followup_answers"],
                idx=idx,
            )
            answer = st.session_state["draft_followup_answers"].get(idx, "")
        else:
            answer = st.text_area("짧게 답하기", value=current_value, height=150, key=answer_key)

        cols = st.columns([1, 1])
        if cols[0].button("건너뛰고 다음 질문", use_container_width=True):
            st.session_state["draft_followup_answers"][idx] = ""
            advance_question()
            st.rerun()
        if cols[1].button("남기고 다음", type="primary", use_container_width=True):
            st.session_state["draft_followup_answers"][idx] = answer.strip()
            advance_question()
            st.rerun()
        return

    question = DEFAULT_QUESTIONS[idx]
    if input_mode == "voice":
        render_voice_question(
            question,
            config=config,
            key=f"draft_base_tts_{idx}",
        )
    else:
        st.markdown(
            f"""
            <div class="question-card">
              <div class="step-label">{idx + 1} / {total}</div>
              <div class="question-text">{escape(question)}</div>
            </div>
                """,
            unsafe_allow_html=True,
        )

    answer_key = f"base_input_{idx}"
    current_value = st.session_state["draft_base_answers"].get(idx, "")
    if input_mode == "voice":
        render_voice_answer_input(
            config,
            key_prefix="base",
            target=st.session_state["draft_base_answers"],
            idx=idx,
        )
        answer = st.session_state["draft_base_answers"].get(idx, "")
        st.caption("정리해서 말하지 않아도 됩니다. 떠오르는 단어만 말해도 괜찮아요.")
    else:
        answer = st.text_area("여기에 짧게 적어보세요", value=current_value, height=170, key=answer_key)
        st.caption("키워드만 남겨도 됩니다.")

    cols = st.columns([1, 1, 1])
    if cols[0].button("이전", use_container_width=True, disabled=idx == 0):
        st.session_state["question_index"] = max(0, idx - 1)
        st.rerun()
    if cols[1].button("건너뛰기", use_container_width=True):
        st.session_state["draft_base_answers"][idx] = ""
        advance_question()
        st.rerun()
    if cols[2].button("다음", type="primary", use_container_width=True):
        st.session_state["draft_base_answers"][idx] = answer.strip()
        with st.spinner("답변해주신 감상을 살펴보고 있어요..."):
            if maybe_offer_followup(llm, idx, answer):
                st.session_state["show_followup"] = True
            else:
                advance_question()
        st.rerun()


def render_free_text_screen(llm: BaseLLMClient | None, config: AppConfig) -> None:
    if llm is None:
        st.warning("LLM provider 설정을 확인하세요.")
        return
    if not restore_draft_work_fields():
        render_missing_draft_title()
        return

    render_record_flow_nav("free_text", title="감상 인터뷰", subtitle="자유 단상")
    st.header("마지막으로 더 남길 단상이 있나요?")
    st.text_area(
        "자유 단상",
        value=st.session_state.get("draft_free_text", ""),
        height=180,
        key="free_text_input",
        placeholder="질문에 들어가지 않은 장면, 문장, 감정, 키워드를 적어도 됩니다.",
    )
    render_audio_input_section(config, key_prefix="free")

    cols = st.columns([1, 2])
    if cols[0].button("질문으로 돌아가기", use_container_width=True):
        st.session_state["question_index"] = len(DEFAULT_QUESTIONS) - 1
        set_screen("question")
        st.rerun()

    if cols[1].button("감상 메모 초안 만들기", type="primary", use_container_width=True):
        st.session_state["draft_free_text"] = st.session_state.get("free_text_input", "").strip()
        try:
            reflection_input = build_reflection_input_from_state()
            answers = collect_all_answers_from_state()
            validate_reflection_input(reflection_input, answers)
        except ValueError as exc:
            st.error(str(exc))
            return

        placeholder = st.empty()
        placeholder.markdown(
            """
            <div class="skeleton-card">
              <div class="memo-label">감상을 정리하고 있어요</div>
              <div class="skeleton-line short"></div>
              <div class="skeleton-line mid"></div>
              <div class="skeleton-line"></div>
              <div class="skeleton-line mid"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            memo = build_structured_memo(llm, reflection_input, answers)
        except LLMClientError as exc:
            placeholder.empty()
            st.error(str(exc))
            return
        placeholder.empty()
        st.session_state["memo"] = memo
        set_screen("edit")
        st.rerun()


def render_edit_screen(provider: str, config: AppConfig) -> None:
    memo = st.session_state.get("memo")
    if not memo:
        set_screen("home")
        st.rerun()
        return

    try:
        reflection_input = build_reflection_input_from_state()
    except ValueError:
        render_missing_draft_title()
        return
    render_record_flow_nav("edit", title="감상 메모 초안", subtitle="수정 후 저장")
    render_memo_editor(memo, key_prefix="new_edit")

    cols = st.columns([1, 2])
    if cols[0].button("자유 단상으로", use_container_width=True):
        set_screen("free")
        st.rerun()

    if cols[1].button("저장하기", type="primary", use_container_width=True):
        answers = collect_all_answers_from_state()
        edited_memo = read_edited_memo(reflection_input, key_prefix="new_edit")
        try:
            record_id = save_record(
                reflection_input=reflection_input,
                questions=[*DEFAULT_QUESTIONS, *st.session_state["draft_followups"].values()],
                answers=answers,
                memo=memo,
                edited_memo=edited_memo,
                provider=provider,
                work_id=st.session_state.get("draft_work_id"),
                force_new_work=bool(st.session_state.get("draft_force_new_work")),
                input_mode=st.session_state.get("draft_input_mode", "text"),
                appreciation_stage=st.session_state.get("draft_appreciation_stage", ""),
                audio_segments=st.session_state.get("draft_audio_segments", []),
                external_context=st.session_state.get("draft_external_context") or {},
            )
            persist_draft_assets(record_id)
            try:
                index_record_embedding(record_id, config)
            except LocalAIError as exc:
                st.session_state["post_save_notice"] = f"저장은 완료됐지만 의미 검색 색인은 만들지 못했습니다: {exc}"
        except Exception as exc:  # pragma: no cover - Streamlit user-facing fallback
            st.error(f"저장 실패: {exc}")
            return
        st.session_state["last_saved_id"] = record_id
        st.session_state["selected_record_id"] = record_id
        set_screen("detail")
        st.rerun()


def render_detail_screen(config: AppConfig, llm: BaseLLMClient | None) -> None:
    record_id = st.session_state.get("selected_record_id")
    record = get_record(record_id) if record_id else None
    if not record:
        st.error("기록을 찾을 수 없습니다.")
        if st.button("홈으로"):
            set_screen("home")
            st.rerun()
        return

    notice = st.session_state.pop("post_save_notice", "")
    if notice:
        st.warning(notice)

    render_home_back_nav("detail", title="기록 상세", subtitle=str(record.get("title") or ""))

    created_at = datetime.fromisoformat(record["created_at"]).strftime("%Y.%m.%d %H:%M")
    updated_at = record.get("updated_at") or record["created_at"]
    updated_label = datetime.fromisoformat(updated_at).strftime("%Y.%m.%d %H:%M")
    mode_label = INPUT_MODE_LABELS.get(str(record.get("input_mode") or "text"), "텍스트")
    meta = [record["content_type"], record["status"], mode_label, f"생성 {created_at}"]
    if record.get("appreciation_stage"):
        meta.insert(2, str(record["appreciation_stage"]))
    if record.get("appreciation_date"):
        meta.insert(2, f"감상일 {record['appreciation_date']}")
    if updated_label != created_at:
        meta.append(f"수정 {updated_label}")

    st.markdown(
        f"""
        <div class="detail-head">
          <div class="record-title">{escape(str(record["title"]))}</div>
          <div class="record-meta">{escape(" · ".join(meta))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    memo = record["edited_memo"] or record["memo"]
    if str(record.get("work_context") or "").strip():
        with st.expander("작품 맥락"):
            st.write(record["work_context"])
    external_context = record.get("external_context") or {}
    if has_external_context(external_context):
        label = external_context_source_label(external_context)
        with st.expander("참고한 작품 정보"):
            page_url = str(external_context.get("page_url") or "").strip()
            st.caption(label)
            if page_url:
                st.markdown(f"[출처 열기]({page_url})")
            st.write(
                str(external_context.get("extract") or "")[
                    : config.wikipedia_prompt_max_chars
                ]
            )
    is_editing = st.session_state.get("detail_editing_record_id") == record["id"]
    if is_editing:
        edit_key_prefix = f"detail_edit_{record['id']}"
        render_memo_editor(memo, key_prefix=edit_key_prefix, title="감상 글 수정")
        edit_cols = st.columns([1, 2])
        if edit_cols[0].button("취소", key=f"cancel_detail_edit_{record['id']}", use_container_width=True):
            st.session_state["detail_editing_record_id"] = None
            st.rerun()
        if edit_cols[1].button("수정사항 저장", type="primary", key=f"save_detail_edit_{record['id']}", use_container_width=True):
            try:
                reflection_input = reflection_input_from_record(record)
                edited_memo = read_edited_memo(reflection_input, key_prefix=edit_key_prefix)
                update_record(record_id=record["id"], edited_memo=edited_memo)
                try:
                    index_record_embedding(record["id"], config)
                except LocalAIError as exc:
                    st.warning(f"수정사항은 저장됐지만 의미 검색 색인은 갱신하지 못했습니다: {exc}")
            except Exception as exc:  # pragma: no cover - Streamlit user-facing fallback
                st.error(f"업데이트 실패: {exc}")
                return
            st.session_state["detail_editing_record_id"] = None
            st.success("수정사항을 저장했습니다.")
            st.rerun()
    else:
        top_actions = st.columns([1, 1, 1])
        if top_actions[0].button("수정하기", key=f"start_detail_edit_{record['id']}", use_container_width=True):
            st.session_state["detail_editing_record_id"] = record["id"]
            st.session_state["delete_confirm_record_id"] = None
            st.rerun()
        work_for_action = get_work(int(record["work_id"])) if record.get("work_id") else None
        if top_actions[1].button("이어 기록", type="primary", key=f"continue_from_detail_top_{record['id']}", use_container_width=True, disabled=work_for_action is None):
            if work_for_action:
                st.session_state["delete_confirm_record_id"] = None
                start_record_for_work(work_for_action)
                st.rerun()
        if top_actions[2].button("삭제하기", key=f"request_delete_{record['id']}", use_container_width=True):
            st.session_state["delete_confirm_record_id"] = record["id"]
            st.rerun()
        if st.session_state.get("delete_confirm_record_id") == record["id"]:
            st.warning("이 감상 글을 삭제할까요? 삭제하면 홈, 검색, 연결된 감상에서 사라집니다.")
            delete_cols = st.columns([1, 1])
            if delete_cols[0].button("삭제 취소", key=f"cancel_delete_{record['id']}", use_container_width=True):
                st.session_state["delete_confirm_record_id"] = None
                st.rerun()
            if delete_cols[1].button("영구 삭제", key=f"confirm_delete_{record['id']}", type="primary", use_container_width=True):
                deleted = delete_record(record["id"])
                st.session_state["selected_record_id"] = None
                st.session_state["selected_work_id"] = None
                st.session_state["detail_editing_record_id"] = None
                st.session_state["delete_confirm_record_id"] = None
                title = str((deleted or {}).get("title") or record.get("title") or "기록")
                st.session_state["post_delete_notice"] = f"'{title}' 감상 글을 삭제했습니다."
                set_screen("home")
                st.rerun()
        render_memo(memo)
        render_record_versions(record["id"])
    if str(record.get("audio_transcript") or "").strip():
        with st.expander("음성 단상 전사"):
            st.write(record["audio_transcript"])
            audio_assets = list_record_assets(record["id"], asset_type="input_audio")
            if audio_assets and Path(str(audio_assets[0].get("path"))).exists():
                st.audio(str(audio_assets[0]["path"]))
    with st.spinner("비슷한 감상을 찾고, 이유를 정리하고 있어요..."):
        similar_cards = []
        for similar in similar_records_for_detail(record, config)[:3]:
            similar_detail = get_record(similar["id"]) or similar
            reason = similarity_reason_for_record_pair(record, similar_detail, llm)
            similar_cards.append((similar, reason))

    if similar_cards:
        st.subheader("연결된 감상")
        st.caption("이 감상과 비슷한 과거 기록이에요.")
        for similar, reason in similar_cards:
            meta_items = [str(similar.get("content_type") or "미지정")]
            meta_items.append("비슷한 감정 단서")
            st.markdown(
                f"""
                <div class="record-card">
                  <div class="record-title">{escape(str(similar["title"]))}</div>
                  <div class="record-meta">{escape(" · ".join(meta_items))}</div>
                  <div class="similar-reason"><strong>비슷한 이유</strong><br>{escape(reason)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("열기", key=f"open_similar_{similar['id']}", use_container_width=True):
                st.session_state["selected_record_id"] = similar["id"]
                set_screen("detail")
                st.rerun()

    with st.expander("질문과 답변"):
        answers = record["answers"]
        for idx, question in enumerate(record["questions"]):
            st.markdown(f"**Q. {question}**")
            st.write(answers.get(question) or "건너뜀")
            render_question_audio_controls(
                question,
                config=config,
                key=f"detail_tts_{record['id']}_{idx}",
                record_id=record["id"],
            )

    cols = st.columns([1, 1, 1])
    if cols[0].button("홈으로", use_container_width=True):
        set_screen("home")
        st.rerun()
    work = get_work(int(record["work_id"])) if record.get("work_id") else None
    if cols[1].button("작품 글", use_container_width=True, disabled=work is None):
        if work:
            st.session_state["selected_work_id"] = work["id"]
            set_screen("work_detail")
            st.rerun()
    if cols[2].button("이어 기록", type="primary", use_container_width=True, disabled=work is None):
        if work:
            start_record_for_work(work)
            st.rerun()


def render_work_card(work: dict[str, Any], *, key_prefix: str, show_add: bool = False) -> None:
    summary = work_summary(work)
    meta = [
        str(work.get("content_type") or "미지정"),
        str(work.get("status") or "미지정"),
        "감상 글",
    ]
    if work.get("_semantic_score") is not None:
        meta.append("의미 검색 결과")
    st.markdown(
        f"""
        <div class="work-card">
          <div class="record-title">{escape(str(work.get("title") or ""))}</div>
          <div class="record-meta">{escape(" · ".join(meta))}</div>
          <div class="record-summary">“{escape(summary)}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1] if show_add else [1])
    if cols[0].button("열기", key=f"{key_prefix}_open_{work['id']}", use_container_width=True):
        record_id = work.get("latest_record_id")
        if record_id:
            st.session_state["selected_record_id"] = int(record_id)
            set_screen("detail")
        else:
            st.session_state["selected_work_id"] = work["id"]
            set_screen("work_detail")
        st.rerun()
    if show_add and cols[1].button(
        "이어 기록",
        key=f"{key_prefix}_add_{work['id']}",
        type="primary",
        use_container_width=True,
    ):
        start_record_for_work(work)
        st.rerun()


def render_home_v2(config: AppConfig) -> None:
    notice = str(st.session_state.pop("post_delete_notice", "") or "").strip()
    st.markdown(
        """
        <div class="drawer-hero">
          <div class="drawer-title">내 감상 서랍</div>
          <div class="drawer-subtitle">작품마다 하나씩 정리한 감상 글을 다시 꺼내보세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if notice:
        st.success(notice)

    search_cols = st.columns([3.2, 1.3, 1.45], gap="small", vertical_alignment="bottom")
    with search_cols[0]:
        search_query = st.text_input(
            "검색",
            placeholder="작품명, 장면, 감정, 기억나는 말로 검색",
            key="home_search",
            label_visibility="collapsed",
        )
    with search_cols[1]:
        search_mode = st.selectbox(
            "검색 방식",
            SEARCH_MODES,
            key="home_search_mode",
            label_visibility="collapsed",
        )
    with search_cols[2]:
        if st.button("+ 새 감상 기록", type="primary", use_container_width=True):
            reset_draft()
            set_screen("info")
            st.rerun()

    selected_content_types = st.session_state.get("home_content_filters", []) or []
    selected_statuses = st.session_state.get("home_status_filters", []) or []
    sort_label = st.session_state.get("home_sort", next(iter(SORT_OPTIONS)))
    keyword_query = search_query if search_mode in {"키워드", "통합"} else ""

    works = list_works(
        keyword_query,
        content_type=selected_content_types,
        status=selected_statuses,
        sort_by=SORT_OPTIONS[sort_label],
    )
    if search_query.strip() and search_mode in {"의미", "통합"}:
        try:
            semantic_records = semantic_record_results(
                search_query,
                config=config,
                content_type=selected_content_types,
                status=selected_statuses,
            )
            semantic_works = works_from_semantic_records(semantic_records)
        except LocalAIDependencyError as exc:
            st.warning(str(exc))
            semantic_works = []
        except LocalAIError as exc:
            st.error(str(exc))
            semantic_works = []
        works = semantic_works if search_mode == "의미" else merge_works(semantic_works, works)

    header_cols = st.columns([4.6, 1.4], gap="medium", vertical_alignment="bottom")
    with header_cols[0]:
        st.markdown(
            f"""
            <div class="list-header">
              <div class="list-title">작품</div>
              <div class="list-subtitle">{len(works)}개의 작품</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown('<div class="filter-label">정렬</div>', unsafe_allow_html=True)
        st.selectbox(
            "정렬",
            list(SORT_OPTIONS),
            key="home_sort",
            label_visibility="collapsed",
        )

    type_cols = st.columns([0.85, 5.15], gap="small", vertical_alignment="center")
    with type_cols[0]:
        st.markdown('<div class="inline-filter-label">유형</div>', unsafe_allow_html=True)
    with type_cols[1]:
        st.pills(
            "콘텐츠 유형",
            CONTENT_FILTERS[1:],
            selection_mode="multi",
            key="home_content_filters",
            width="stretch",
            label_visibility="collapsed",
        )

    status_cols = st.columns([0.85, 5.15], gap="small", vertical_alignment="center")
    with status_cols[0]:
        st.markdown('<div class="inline-filter-label">상태</div>', unsafe_allow_html=True)
    with status_cols[1]:
        st.pills(
            "감상 상태",
            STATUS_FILTERS[1:],
            selection_mode="multi",
            key="home_status_filters",
            width="stretch",
            label_visibility="collapsed",
        )

    st.markdown('<div class="toolbar-divider"></div>', unsafe_allow_html=True)

    if not works:
        if search_query.strip() or selected_content_types or selected_statuses:
            render_empty_state(
                title="검색 결과가 없어요.",
                body="다른 작품명이나 감정 키워드로 검색해보세요. 예: 찝찝한 결말, 외로움, 다시 읽기",
                button_label="전체 작품 보기",
                button_key="clear_home_filters_v2",
                on_click=clear_home_filters,
            )
            return
        if render_empty_state(
            title="아직 남긴 감상이 없어요.",
            body="질문에 짧게 답하면 작품마다 다시 읽기 좋은 감상 글이 만들어집니다.",
            button_label="첫 감상 기록하기",
            button_key="first_work_record",
        ):
            reset_draft()
            set_screen("info")
            st.rerun()
        return

    for work in works[:20]:
        render_work_card(work, key_prefix="home", show_add=False)


def render_work_match_screen() -> None:
    title = st.session_state.get("draft_title", "").strip()
    content_type = st.session_state.get("draft_content_type")
    matches = find_matching_works(title, content_type)
    if not matches:
        set_screen("mode")
        st.rerun()
        return

    render_record_flow_nav(
        "work_match",
        title="기존 작품 확인",
        subtitle=title,
        show_info_back=True,
    )
    st.header("이미 기록한 작품이 있어요")
    st.markdown('<p class="quiet-caption">같은 작품이라면 새 글을 만들지 않고 기존 감상 글을 이어서 업데이트합니다.</p>', unsafe_allow_html=True)
    render_draft_external_context_status()
    for work in matches:
        render_work_card(work, key_prefix="match", show_add=False)
        cols = st.columns([1, 1])
        if cols[0].button("기존 글 업데이트", key=f"match_add_{work['id']}", type="primary", use_container_width=True):
            start_record_for_work(
                work,
                external_context=st.session_state.get("draft_external_context") or {},
            )
            st.rerun()
        if cols[1].button("새 작품으로 만들기", key=f"match_new_{work['id']}", use_container_width=True):
            st.session_state["draft_work_id"] = None
            st.session_state["draft_force_new_work"] = True
            set_screen("mode")
            st.rerun()

def render_mode_screen() -> None:
    if not restore_draft_work_fields():
        render_missing_draft_title()
        return

    render_record_flow_nav(
        "mode",
        title="기록 방식 선택",
        subtitle=str(st.session_state.get("draft_title") or ""),
        show_info_back=True,
    )
    st.header("어떤 방식으로 기록할까요?")
    st.markdown('<p class="quiet-caption">텍스트와 음성은 같은 감상 인터뷰 흐름을 사용합니다. 입력 방식만 달라집니다.</p>', unsafe_allow_html=True)
    render_draft_external_context_status()
    cols = st.columns(2, gap="medium")
    with cols[0]:
        st.markdown(
            """
            <div class="mode-card">
              <div class="record-title">텍스트로 기록하기</div>
              <div class="record-meta">질문에 짧게 답하며 감상을 정리해요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("텍스트로 시작", key="start_text_mode", type="primary", use_container_width=True):
            st.session_state["draft_input_mode"] = "text"
            set_screen("question")
            st.rerun()
    with cols[1]:
        st.markdown(
            """
            <div class="mode-card">
              <div class="record-title">음성으로 기록하기</div>
              <div class="record-meta">음성 파일을 전사해 같은 질문 흐름에 사용해요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("음성으로 시작", key="start_voice_mode", use_container_width=True):
            st.session_state["draft_input_mode"] = "voice"
            set_screen("question")
            st.rerun()

def render_work_detail_screen(config: AppConfig) -> None:
    work_id = st.session_state.get("selected_work_id")
    work = get_work(int(work_id)) if work_id else None
    if not work:
        st.error("작품을 찾을 수 없습니다.")
        if st.button("홈으로"):
            set_screen("home")
            st.rerun()
        return

    render_home_back_nav("work_detail", title="작품 글", subtitle=str(work.get("title") or ""))

    record_id = work.get("latest_record_id")
    record = get_record(int(record_id)) if record_id else None
    meta = [
        str(work.get("content_type") or "미지정"),
        str(work.get("status") or "미지정"),
        "감상 글 1개",
    ]
    st.markdown(
        f"""
        <div class="detail-head">
          <div class="record-title">{escape(str(work["title"]))}</div>
          <div class="record-meta">{escape(" · ".join(meta))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    add_cols = st.columns([1, 1])
    if add_cols[0].button("텍스트로 이어 기록", type="primary", use_container_width=True):
        start_record_for_work(work, input_mode="text")
        st.rerun()
    if add_cols[1].button("음성으로 이어 기록", use_container_width=True):
        start_record_for_work(work, input_mode="voice")
        st.rerun()

    st.subheader("감상 글")
    if not record:
        st.info("아직 이 작품에 저장된 감상 글이 없습니다.")
        return
    memo = normalize_memo(record.get("edited_memo") or record.get("memo") or {})
    summary = str(memo.get("one_line_summary") or record.get("initial_note") or "요약 없음")
    date_label = str(record.get("appreciation_date") or record.get("updated_at") or record.get("created_at") or "")[:10]
    stage = str(record.get("appreciation_stage") or "").strip()
    mode = INPUT_MODE_LABELS.get(str(record.get("input_mode") or "text"), "텍스트")
    st.markdown(
        f"""
        <div class="timeline-item">
          <div class="timeline-date">{escape(date_label)}{escape(" · " + stage if stage else "")} · {escape(mode)}</div>
          <div class="record-summary">“{escape(summary)}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("감상 글 열기", key=f"open_work_record_{record['id']}", use_container_width=True):
        st.session_state["selected_record_id"] = record["id"]
        set_screen("detail")
        st.rerun()


def render_current_screen(provider: str, llm: BaseLLMClient | None, config: AppConfig) -> None:
    screen = st.session_state.get("screen", "home")
    if screen == "home":
        render_home_v2(config)
    elif screen == "info":
        render_info_screen(config)
    elif screen == "work_match":
        render_work_match_screen()
    elif screen == "mode":
        render_mode_screen()
    elif screen == "work_detail":
        render_work_detail_screen(config)
    elif screen == "question":
        render_question_screen(llm, config)
    elif screen == "free":
        render_free_text_screen(llm, config)
    elif screen == "edit":
        render_edit_screen(provider, config)
    elif screen == "detail":
        render_detail_screen(config, llm)
    else:
        set_screen("home")
        st.rerun()


def main() -> None:
    config = AppConfig.from_env()
    st.set_page_config(page_title=config.app_name, layout="centered")
    apply_theme()
    ensure_state()
    init_db()

    provider, llm = render_provider_selector(config)
    render_current_screen(provider, llm, config)


if __name__ == "__main__":
    main()
