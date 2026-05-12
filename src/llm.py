from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
from openai import OpenAI

from src.config import AppConfig
from src.models import ReflectionInput
from src.prompts import build_memo_prompt, build_question_prompt


class LLMClientError(RuntimeError):
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_questions(self, reflection_input: ReflectionInput) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def generate_questions(self, reflection_input: ReflectionInput) -> list[str]:
        title = reflection_input.title
        status = reflection_input.status
        return [
            f"{title}에서 가장 오래 기억하고 싶은 장면이나 문장은 무엇인가요?",
            "그 장면을 보거나 읽었을 때 어떤 감정이 먼저 들었나요?",
            "이 작품을 보며 떠오른 다른 작품, 경험, 사람, 생각이 있나요?",
            "나중에 다시 이어 보거나 떠올릴 때 꼭 기억하고 싶은 맥락은 무엇인가요?",
            f"{status} 상태의 지금, 아직 더 생각해보고 싶은 질문은 무엇인가요?",
        ]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        answered = [answer for answer in answers.values() if answer.strip()]
        note = reflection_input.initial_note
        answer_text = " ".join(answered)

        return {
            "one_line_summary": f"{reflection_input.title}에 대한 현재 감상을 다시 꺼내보기 위한 기록",
            "content_summary": note,
            "impressive_points": answered[:2] or [note],
            "personal_interpretation": answer_text or "아직 해석이 충분히 정리되지 않았습니다.",
            "emotional_trace": answered[1] if len(answered) > 1 else note,
            "revisit_points": answered[-1] if answered else "다음 감상 때 더 구체적으로 기록해볼 지점이 남아 있습니다.",
            "comparison_points": answered[2] if len(answered) > 2 else "비교 대상은 아직 기록되지 않았습니다.",
            "review_note": (
                f"{reflection_input.status} 상태에서 남긴 기록입니다. "
                f"다시 볼 때는 '{note[:80]}'에서 시작하면 당시 감상을 빠르게 복원할 수 있습니다."
            ),
            "tags": [reflection_input.content_type, reflection_input.status, reflection_input.title],
        }


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMClientError("OPENAI_API_KEY가 없어 OpenAI provider를 사용할 수 없습니다.")
        self.client = OpenAI()
        self.model = model

    def generate_questions(self, reflection_input: ReflectionInput) -> list[str]:
        content = self._complete_json(build_question_prompt(reflection_input))
        questions = content.get("questions")
        if not isinstance(questions, list):
            raise LLMClientError("질문 생성 결과 형식이 올바르지 않습니다.")
        return [str(question).strip() for question in questions if str(question).strip()][:5]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        content = self._complete_json(build_memo_prompt(reflection_input, answers))
        if not isinstance(content, dict):
            raise LLMClientError("메모 생성 결과 형식이 올바르지 않습니다.")
        return content

    def _complete_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        return _parse_json(response.output_text)


class OllamaLLMClient(BaseLLMClient):
    def __init__(self, model: str, url: str) -> None:
        self.model = model
        self.url = url

    def generate_questions(self, reflection_input: ReflectionInput) -> list[str]:
        content = self._complete_json(build_question_prompt(reflection_input))
        questions = content.get("questions")
        if not isinstance(questions, list):
            raise LLMClientError("질문 생성 결과 형식이 올바르지 않습니다.")
        return [str(question).strip() for question in questions if str(question).strip()][:5]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        return self._complete_json(build_memo_prompt(reflection_input, answers))

    def _complete_json(self, prompt: str) -> dict[str, Any]:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"Ollama 요청 실패: {exc}") from exc

        payload = response.json()
        return _parse_json(payload.get("response", ""))


def create_llm_client(provider: str, config: AppConfig) -> BaseLLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        return OpenAILLMClient(config.openai_model)
    if provider == "ollama":
        return OllamaLLMClient(config.ollama_model, config.ollama_url)
    raise LLMClientError(f"지원하지 않는 LLM provider입니다: {provider}")


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"LLM 응답을 JSON으로 해석할 수 없습니다: {text[:200]}") from exc

    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON 응답은 object여야 합니다.")
    return parsed
