from __future__ import annotations

import hashlib
import io
import math
import re
import wave
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    AppConfig,
    AUDIO_UPLOAD_DIR,
    TTS_ASSET_DIR,
)


class LocalAIError(RuntimeError):
    pass


class LocalAIDependencyError(LocalAIError):
    pass


EMBEDDING_TEXT_MAX_CHARS = 2_000
EMBEDDING_MAX_SEQ_LENGTH = 512


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    audio_path: str


@dataclass(frozen=True)
class AudioStats:
    duration_seconds: float
    rms: float
    peak: int


SUBTITLE_HALLUCINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(?:한글\s*)?자막\s*(?:제공\s*)?(?:by|제작|제작자|번역)?\s*[:：]?\s*[\w가-힣 ._-]{0,40}\s*$",
        r"^\s*(?:subtitles?|caption(?:s|ing)?)\s*(?:by|provided by|created by)?\s*[:：]?\s*[\w가-힣 ._-]{0,40}\s*$",
        r"^\s*자막\s*제공\s*및\s*자막을\s*사용하였습니다\.?\s*$",
        r"^\s*시청해\s*주셔서\s*감사합니다\.?\s*$",
    ]
]


def safe_stem(value: str) -> str:
    stem = Path(value).stem.strip().lower()
    stem = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "-", stem)
    return stem.strip("-") or "asset"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_uploaded_audio(file_name: str, data: bytes) -> Path:
    AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file_name).suffix.lower() or ".wav"
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = AUDIO_UPLOAD_DIR / f"{safe_stem(file_name)}-{digest}{suffix}"
    path.write_bytes(data)
    return path


def _stats_from_wav_reader(wav_file: wave.Wave_read) -> AudioStats:
    sample_width = wav_file.getsampwidth()
    frame_count = wav_file.getnframes()
    frame_rate = wav_file.getframerate() or 1
    frames = wav_file.readframes(frame_count)

    if not frames:
        return AudioStats(duration_seconds=0.0, rms=0.0, peak=0)

    if sample_width == 1:
        samples = [abs(byte - 128) for byte in frames]
    elif sample_width == 2:
        samples_16 = array("h")
        samples_16.frombytes(frames)
        samples = [abs(int(sample)) for sample in samples_16]
    elif sample_width == 4:
        samples_32 = array("i")
        samples_32.frombytes(frames)
        samples = [abs(int(sample)) for sample in samples_32]
    else:
        return AudioStats(duration_seconds=frame_count / frame_rate, rms=0.0, peak=0)

    peak = max(samples) if samples else 0
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) if samples else 0.0
    return AudioStats(duration_seconds=frame_count / frame_rate, rms=rms, peak=peak)


def wav_audio_stats(path: str | Path) -> AudioStats | None:
    audio_path = Path(path)
    if audio_path.suffix.lower() != ".wav":
        return None

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            return _stats_from_wav_reader(wav_file)
    except (wave.Error, OSError):
        return None


def wav_audio_stats_from_bytes(data: bytes) -> AudioStats | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            return _stats_from_wav_reader(wav_file)
    except (wave.Error, OSError):
        return None


def is_probably_silent_wav(path: str | Path, *, peak_threshold: int = 8) -> bool:
    stats = wav_audio_stats(path)
    if stats is None:
        return False

    return stats.peak <= peak_threshold


def clean_transcript_text(text: str) -> str:
    lines = []
    for raw_line in re.split(r"[\r\n]+", text):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if any(pattern.match(line) for pattern in SUBTITLE_HALLUCINATION_PATTERNS):
            continue
        line = re.sub(
            r"(?:\s*(?:한글\s*)?자막\s*(?:by|제작|제작자|번역)?\s*[:：]?\s*[\w가-힣 ._-]{0,40})$",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def transcribe_audio_file(path: str | Path, config: AppConfig) -> TranscriptionResult:
    if is_probably_silent_wav(path):
        raise LocalAIError(
            "녹음에 목소리가 들어오지 않았어요. 브라우저 마이크 권한, 입력 장치, 음소거 상태를 확인한 뒤 다시 녹음해 주세요."
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise LocalAIDependencyError(
            "faster-whisper가 설치되어 있지 않습니다. `pip install -r requirements-local-ai.txt` 후 다시 시도하세요."
        ) from exc

    model = _get_whisper_model(
        config.stt_model,
        config.stt_device,
        config.stt_compute_type,
    )
    try:
        segments, _info = model.transcribe(
            str(path),
            language="ko",
            vad_filter=True,
        )
        segment_items = []
        for segment in segments:
            cleaned_text = clean_transcript_text(segment.text or "")
            if not cleaned_text:
                continue
            segment_items.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": cleaned_text,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on local model/runtime
        raise LocalAIError(f"음성 전사 실패: {exc}") from exc

    text = " ".join(item["text"] for item in segment_items).strip()
    return TranscriptionResult(text=text, segments=segment_items, audio_path=str(path))


@lru_cache(maxsize=2)
def _get_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def embed_texts(texts: list[str], config: AppConfig) -> tuple[str, list[list[float]]]:
    if not texts:
        return config.embedding_model, []

    prepared_texts = [truncate_embedding_text(text) for text in texts]

    try:
        model = _get_embedding_model(config.embedding_model)
        vectors = _encode_with_sentence_transformer(model, prepared_texts)
        return config.embedding_model, vectors
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise LocalAIError(f"임베딩 생성 실패: {exc}") from exc
        _clear_cuda_cache()
        try:
            model = _get_embedding_model(config.embedding_fallback_model, device="cpu")
            vectors = _encode_with_sentence_transformer(model, prepared_texts)
            return config.embedding_fallback_model, vectors
        except Exception as fallback_exc:  # pragma: no cover - depends on local model/runtime
            raise LocalAIError(f"Embedding fallback failed: {fallback_exc}") from fallback_exc
    except ImportError as exc:
        raise LocalAIDependencyError(
            "sentence-transformers가 설치되어 있지 않습니다. `pip install -r requirements-local-ai.txt` 후 다시 시도하세요."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on local model/runtime
        raise LocalAIError(f"임베딩 생성 실패: {exc}") from exc


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str, device: str | None = None):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=resolved_device, trust_remote_code=True)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = min(int(model.max_seq_length), EMBEDDING_MAX_SEQ_LENGTH)
    return model


def truncate_embedding_text(text: str, max_chars: int = EMBEDDING_TEXT_MAX_CHARS) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = max_chars // 2
    tail = max_chars - head
    return f"{cleaned[:head]} ... {cleaned[-tail:]}"


def _encode_with_sentence_transformer(model: Any, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in row] for row in np.asarray(vectors)]


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def stable_text_embedding(text: str, *, dimensions: int = 64) -> list[float]:
    """Small deterministic embedding used by tests and offline smoke checks."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"[\w가-힣]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.tolist()
    return (vector / norm).astype(float).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0 or math.isnan(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_embedding_matches(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        score = cosine_similarity(query_embedding, candidate.get("embedding") or [])
        if score <= 0:
            continue
        item = dict(candidate)
        item["score"] = score
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def record_embedding_text(record: dict[str, Any], memo: dict[str, Any]) -> str:
    parts: list[str] = [
        str(record.get("title") or ""),
        str(record.get("content_type") or ""),
        str(record.get("status") or ""),
        str(record.get("work_context") or ""),
        str(record.get("spoiler_scope") or ""),
        str(record.get("initial_note") or ""),
        str(record.get("free_text") or ""),
        str(record.get("audio_transcript") or ""),
    ]
    for value in memo.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    answers = record.get("answers") or {}
    parts.extend(str(value) for value in answers.values())
    return "\n".join(part.strip() for part in parts if part and str(part).strip())


def tts_output_path(text: str, config: AppConfig, *, record_id: int | None = None) -> Path:
    TTS_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"record-{record_id}" if record_id is not None else "draft"
    digest = text_hash(f"{config.tts_model}|{config.tts_speaker}|{text}")[:16]
    return TTS_ASSET_DIR / f"{prefix}-question-{digest}.wav"


def generate_question_audio(text: str, output_path: str | Path, config: AppConfig) -> Path:
    output = Path(output_path)
    if output.exists():
        return output

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise LocalAIDependencyError(
            "Qwen TTS 의존성이 설치되어 있지 않습니다. `pip install -r requirements-local-ai.txt` 후 다시 시도하세요."
        ) from exc

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = _get_tts_model(config.tts_model, device, str(dtype))
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=config.tts_speaker,
            language="Korean",
            do_sample=True,
            temperature=0.7,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), wavs[0], sample_rate)
    except Exception as exc:  # pragma: no cover - depends on local model/runtime
        raise LocalAIError(f"질문 음성 생성 실패: {exc}") from exc
    return output


@lru_cache(maxsize=1)
def _get_tts_model(model_name: str, device: str, dtype_repr: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if "bfloat16" in dtype_repr else torch.float32
    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )
