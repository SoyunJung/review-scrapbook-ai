from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
from openai import OpenAI

from src.config import AppConfig
from src.models import ReflectionInput
from src.prompts import (
    MEMO_JSON_SCHEMA,
    QUESTION_JSON_SCHEMA,
    SIMILARITY_JSON_SCHEMA,
    build_memo_prompt,
    build_question_prompt,
    build_similarity_prompt,
)


class LLMClientError(RuntimeError):
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_questions(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def explain_similarity(
        self,
        source: dict[str, object],
        target: dict[str, object],
    ) -> str:
        return ""


class MockLLMClient(BaseLLMClient):
    def generate_questions(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> list[str]:
        title = reflection_input.title
        answered_text = " ".join(answer for answer in answers.values() if answer)
        questions: list[str] = []
        if "왜" not in answered_text and "이유" not in answered_text:
            questions.append(f"{title}에서 남긴 단상이 왜 중요하게 느껴졌는지 한 단서만 더 붙여볼까요?")
        if "감정" not in answered_text and "느낌" not in answered_text:
            questions.append("그때의 감정은 편안함, 불편함, 아쉬움, 설렘 중 어디에 더 가까웠나요?")
        if "비교" not in answered_text and "떠오" not in answered_text:
            questions.append("이 감상을 다른 작품이나 개인 경험과 연결한다면 무엇이 떠오르나요?")
        return [
            *questions,
            f"{reflection_input.status} 상태에서 나중에 다시 볼 때 필요한 힌트를 한 문장으로 남긴다면 무엇인가요?",
        ][:3]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        answered = [answer for answer in answers.values() if answer.strip()]
        note = (
            reflection_input.initial_note
            or reflection_input.free_text
            or reflection_input.audio_transcript
            or "추가 기록 필요"
        )
        answer_text = " ".join(answered)
        context = reflection_input.work_context or note

        return {
            "title": reflection_input.title,
            "content_type": reflection_input.content_type,
            "appreciation_date": reflection_input.appreciation_date or "입력 없음",
            "status": reflection_input.status,
            "one_line_summary": (
                f"{note[:44]}에서 출발해, 사용자가 남긴 장면과 감정을 다시 읽기 좋은 흐름으로 묶은 "
                f"{reflection_input.title} 감상 기록"
            ),
            "content_context": (
                f"이 기록은 '{context[:140]}'라는 맥락에서 출발한다. "
                "사용자는 작품 전체를 요약하기보다 자신에게 남은 장면, 감정, 이유를 중심으로 감상을 남겼다. "
                "따라서 이 메모는 줄거리 설명보다 나중에 다시 읽었을 때 당시의 감정선을 복원하는 데 초점을 둔다."
            ),
            "impressive_points": answered[:3] or [note],
            "feelings_and_interpretation": (
                f"사용자가 남긴 단서는 다음과 같다. {answer_text} "
                "이 답변들을 종합하면, 감상은 단순한 좋고 싫음보다 특정 장면이 남긴 감정의 결을 복원하는 쪽에 가깝다. "
                "짧은 표현 안에서도 장면, 이유, 비교, 다시 볼 지점이 서로 이어져 있어 하나의 감상 흐름으로 정리할 수 있다. "
                "다만 사용자가 말하지 않은 줄거리나 결말은 덧붙이지 않고, 사용자가 직접 남긴 단서 안에서만 해석을 확장한다."
                if answer_text
                else "아직 해석이 충분히 정리되지 않았습니다."
            ),
            "comparison_points": answered[2] if len(answered) > 2 else "비교 대상은 아직 기록되지 않았습니다.",
            "revisit_points": answered[-1] if answered else "다음 감상 때 더 구체적으로 기록해볼 지점이 남아 있습니다.",
            "review_note": (
                f"{reflection_input.status} 상태에서 남긴 기록입니다. "
                f"다시 볼 때는 '{note[:80]}'에서 시작해, 왜 그 장면이 남았는지와 어떤 감정으로 이어졌는지를 먼저 확인하면 좋습니다."
            ),
            "original_notes": answer_text or note,
            "tags": [reflection_input.content_type, reflection_input.status, reflection_input.title],
        }

    def explain_similarity(
        self,
        source: dict[str, object],
        target: dict[str, object],
    ) -> str:
        source_tags = source.get("tags")
        target_tags = target.get("tags")
        if isinstance(source_tags, list) and isinstance(target_tags, list):
            shared = [str(tag) for tag in source_tags if tag in target_tags][:3]
            if shared:
                return f"두 기록은 {', '.join(shared)} 같은 단서를 공유해요. 감상도 작품 설명보다 사용자가 남긴 감정의 결을 다시 꺼내보게 합니다."
        return "두 기록 모두 작품의 사건보다 사용자가 남긴 감정과 해석의 방향이 서로 닿아 있어 연결해 볼 만합니다."


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMClientError("OPENAI_API_KEY가 없어 OpenAI provider를 사용할 수 없습니다.")
        self.client = OpenAI()
        self.model = model

    def generate_questions(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> list[str]:
        content = self._complete_json(
            build_question_prompt(reflection_input, answers),
            schema_name="followup_questions",
            schema=QUESTION_JSON_SCHEMA,
        )
        questions = content.get("questions")
        if not isinstance(questions, list):
            raise LLMClientError("질문 생성 결과 형식이 올바르지 않습니다.")
        return [str(question).strip() for question in questions if str(question).strip()][:3]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        content = self._complete_json(
            build_memo_prompt(reflection_input, answers),
            schema_name="reflection_memo",
            schema=MEMO_JSON_SCHEMA,
        )
        if not isinstance(content, dict):
            raise LLMClientError("메모 생성 결과 형식이 올바르지 않습니다.")
        return content

    def explain_similarity(
        self,
        source: dict[str, object],
        target: dict[str, object],
    ) -> str:
        content = self._complete_json(
            build_similarity_prompt(source, target),
            schema_name="similarity_reason",
            schema=SIMILARITY_JSON_SCHEMA,
        )
        reason = str(content.get("reason") or "").strip()
        if not reason:
            raise LLMClientError("유사 기록 설명 결과가 비어 있습니다.")
        return reason

    def _complete_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return _parse_json(response.output_text)


class OllamaLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        url: str,
        *,
        think: bool = False,
        timeout_seconds: int = 300,
        num_predict: int = 700,
    ) -> None:
        self.model = model
        self.url = url
        self.think = think
        self.timeout_seconds = timeout_seconds
        self.num_predict = num_predict

    def generate_questions(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> list[str]:
        content = self._complete_json(build_question_prompt(reflection_input, answers))
        questions = content.get("questions")
        if not isinstance(questions, list):
            raise LLMClientError("질문 생성 결과 형식이 올바르지 않습니다.")
        return [str(question).strip() for question in questions if str(question).strip()][:3]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        return self._complete_json(build_memo_prompt(reflection_input, answers))

    def explain_similarity(
        self,
        source: dict[str, object],
        target: dict[str, object],
    ) -> str:
        content = self._complete_json(build_similarity_prompt(source, target))
        reason = str(content.get("reason") or "").strip()
        if not reason:
            raise LLMClientError("유사 기록 설명 결과가 비어 있습니다.")
        return reason

    def _complete_json(self, prompt: str) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt if self.think else f"/no_think\n{prompt}",
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
                "num_predict": self.num_predict,
            },
        }
        if not self.think:
            request_payload["format"] = "json"

        try:
            response = requests.post(
                self.url,
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"Ollama 요청 실패: {exc}") from exc

        payload = response.json()
        response_text = payload.get("response", "")
        if not response_text and self.think:
            response_text = payload.get("thinking", "")
        return _parse_json(response_text)


def create_llm_client(provider: str, config: AppConfig) -> BaseLLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        return OpenAILLMClient(config.openai_model)
    if provider == "ollama":
        return OllamaLLMClient(
            config.ollama_model,
            config.ollama_url,
            think=config.ollama_think,
            timeout_seconds=config.ollama_timeout_seconds,
            num_predict=config.ollama_num_predict,
        )
    raise LLMClientError(f"지원하지 않는 LLM provider입니다: {provider}")


def _parse_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(text)
        if not extracted:
            raise LLMClientError(f"LLM 응답을 JSON으로 해석할 수 없습니다: {text[:200]}") from exc
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as nested_exc:
            raise LLMClientError(f"LLM 응답을 JSON으로 해석할 수 없습니다: {text[:200]}") from nested_exc

    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON 응답은 object여야 합니다.")
    return parsed


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape_next = False
    for index, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""
