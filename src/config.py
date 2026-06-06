from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "scrapbook.db"
UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audio"
ASSET_DIR = DATA_DIR / "assets"
TTS_ASSET_DIR = ASSET_DIR / "tts"


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Reflection Scrapbook AI"
    llm_provider: str = "mock"
    openai_model: str = "gpt-5.5"
    ollama_model: str = "qwen3.5:9b"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_think: bool = False
    ollama_timeout_seconds: int = 600
    ollama_num_predict: int = 3500
    stt_model: str = "large-v3"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8_float16"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_fallback_model: str = "Qwen/Qwen3-Embedding-0.6B"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_speaker: str = "Sohee"
    wikipedia_context_enabled: bool = True
    wikipedia_language: str = "en"
    wikipedia_timeout_seconds: int = 12
    wikipedia_store_max_chars: int = 100_000
    wikipedia_prompt_max_chars: int = 7_000
    wikipedia_user_agent: str = "ReflectionScrapbookAI/0.1 (university project; local development)"

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(ROOT_DIR / ".env")
        provider = os.getenv("LLM_PROVIDER", "mock").lower()
        if provider not in {"mock", "openai", "ollama"}:
            provider = "mock"

        return cls(
            llm_provider=provider,
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            ollama_think=_env_bool("OLLAMA_THINK", cls.ollama_think),
            ollama_timeout_seconds=int(
                os.getenv("OLLAMA_TIMEOUT_SECONDS", str(cls.ollama_timeout_seconds))
            ),
            ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", str(cls.ollama_num_predict))),
            stt_model=os.getenv("STT_MODEL", cls.stt_model),
            stt_device=os.getenv("STT_DEVICE", cls.stt_device),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", cls.stt_compute_type),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_fallback_model=os.getenv(
                "EMBEDDING_FALLBACK_MODEL",
                cls.embedding_fallback_model,
            ),
            tts_model=os.getenv("TTS_MODEL", cls.tts_model),
            tts_speaker=os.getenv("TTS_SPEAKER", cls.tts_speaker),
            wikipedia_context_enabled=_env_bool(
                "WIKIPEDIA_CONTEXT_ENABLED",
                cls.wikipedia_context_enabled,
            ),
            wikipedia_language=os.getenv("WIKIPEDIA_LANGUAGE", cls.wikipedia_language),
            wikipedia_timeout_seconds=int(
                os.getenv(
                    "WIKIPEDIA_TIMEOUT_SECONDS",
                    str(cls.wikipedia_timeout_seconds),
                )
            ),
            wikipedia_store_max_chars=int(
                os.getenv(
                    "WIKIPEDIA_STORE_MAX_CHARS",
                    str(cls.wikipedia_store_max_chars),
                )
            ),
            wikipedia_prompt_max_chars=int(
                os.getenv(
                    "WIKIPEDIA_PROMPT_MAX_CHARS",
                    str(cls.wikipedia_prompt_max_chars),
                )
            ),
            wikipedia_user_agent=os.getenv(
                "WIKIPEDIA_USER_AGENT",
                cls.wikipedia_user_agent,
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
