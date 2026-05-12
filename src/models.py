from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ReflectionInput(BaseModel):
    content_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    creator: str | None = None
    status: str = Field(min_length=1)
    initial_note: str = Field(min_length=3)

    @field_validator("title", "initial_note")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("필수 입력값이 비어 있습니다.")
        return stripped

    @field_validator("creator")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
