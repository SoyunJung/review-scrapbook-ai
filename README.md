# review-scrapbook-ai

문화콘텐츠 감상 직후의 짧고 파편적인 생각을 AI 질문으로 끌어내고, 나중에 다시 읽기 좋은 구조화 감상 메모로 저장하는 개인 감상 아카이브입니다.

## Features

- 작품 단위 감상 아카이브
- 기존 작품 감지 및 이어 기록
- 텍스트/음성 기반 감상 인터뷰
- 기본 질문 5개와 선택형 파생 질문
- AI 구조화 메모 생성, 수정, 저장
- SQLite 기반 로컬 저장
- 키워드 검색, 의미 검색, 연결된 감상 표시
- Wikipedia EN 기반 작품 맥락 참고
- faster-whisper 기반 음성 전사
- Qwen3-TTS 기반 질문 음성 출력

## Project Structure

```text
.
├── app.py
├── src/
├── scripts/
├── tests/
├── requirements.txt
├── requirements-local-ai.txt
├── .env.example
└── .streamlit/
```

- `app.py`: Streamlit 애플리케이션 진입점
- `src/`: DB, LLM, 로컬 AI, 프롬프트, 질문, 작품 맥락 조회 로직
- `scripts/`: 데모 데이터 생성 및 임베딩 색인 재생성 스크립트
- `tests/`: 주요 MVP 흐름과 DB/검색 로직 테스트

## Installation

Python 3.11 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

로컬 AI 기능까지 실행하려면 PyTorch CUDA wheel과 추가 의존성을 설치합니다.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-local-ai.txt
```

Ollama를 설치한 뒤 LLM 모델을 받습니다.

```powershell
winget install Ollama.Ollama
ollama pull qwen3.5:9b
```

모델 파일은 Git 저장소에 포함하지 않습니다. Ollama 저장소와 Hugging Face cache에서 로컬로 관리합니다.

## Configuration

`.env.example`을 `.env`로 복사한 뒤 실행 환경에 맞게 수정합니다.

```powershell
Copy-Item .env.example .env
```

기본 로컬 AI 설정은 다음과 같습니다.

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_URL=http://localhost:11434/api/generate

STT_MODEL=Systran/faster-whisper-large-v3
STT_DEVICE=cuda
STT_COMPUTE_TYPE=int8_float16

EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

WIKIPEDIA_CONTEXT_ENABLED=true
WIKIPEDIA_LANGUAGE=en
```

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
python -m compileall app.py src tests scripts
python -m unittest discover -s tests
```

## AI Components

- LLM: Ollama `qwen3.5:9b`
- STT: `Systran/faster-whisper-large-v3`
- Embedding: `Qwen/Qwen3-Embedding-0.6B`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- External context: Wikipedia EN Action API

이미지 생성은 최종 구현 범위에서 제외했습니다.

## Data Model

핵심 구조는 작품 1개에 대표 감상 글 1개가 연결되는 형태입니다.

```text
work
└── record
```

- `scrapbook_works`: 작품 제목, 유형, 현재 감상 상태
- `scrapbook_records`: 작품별 대표 감상 글, 질문 답변, 생성/수정 메모, 작품 맥락
- `record_embeddings`: 의미 검색과 연결된 감상 추천용 벡터
- `record_assets`: 입력 음성, 질문 TTS 등 로컬 파일 자산

같은 작품으로 이어 기록하면 기존 대표 감상 글이 업데이트됩니다.

## Main Flow

1. 작품 정보 입력
2. 기존 작품 감지
3. 텍스트 또는 음성 감상 인터뷰
4. AI 구조화 메모 생성
5. 사용자의 메모 수정 및 저장
6. 검색, 의미 검색, 연결된 감상 조회

## Notes

- 로컬 DB, 업로드한 음성 파일, 모델 weight/cache, 보고서 PDF, 기획서 원본은 저장소에 포함하지 않습니다.
- 의미 검색 색인은 저장/수정 시 자동 갱신됩니다.
- 전체 임베딩 색인을 다시 만들려면 아래 명령을 사용할 수 있습니다.

```powershell
python scripts/rebuild_embedding_index.py
```
