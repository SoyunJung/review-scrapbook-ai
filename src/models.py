from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ReflectionInput(BaseModel):
    content_type: str = "미지정"
    title: str = Field(min_length=1)
    creator: str | None = None
    appreciation_date: str | None = None
    status: str = "미지정"
    work_context: str = ""
    external_context: str = ""
    external_context_source: str = ""
    spoiler_scope: str = "입력 내용에서 추리한 감상 범위까지만"
    initial_note: str = ""
    free_text: str = ""
    audio_transcript: str = ""

    @field_validator("title")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("콘텐츠 제목은 필수입니다.")
        return stripped

    @field_validator(
        "content_type",
        "status",
        "work_context",
        "external_context",
        "external_context_source",
        "spoiler_scope",
        "initial_note",
        "free_text",
        "audio_transcript",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("creator", "appreciation_date")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
