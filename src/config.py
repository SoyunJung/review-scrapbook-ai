from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "scrapbook.db"


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Reflection Scrapbook AI"
    llm_provider: str = "mock"
    openai_model: str = "gpt-4.1-mini"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434/api/generate"

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
        )
