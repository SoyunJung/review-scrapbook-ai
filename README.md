# review-scrapbook-ai

문화콘텐츠를 감상한 직후 떠오르는 파편적인 생각을 AI 질문으로 끌어내고, 나중에 다시 읽기 좋은 구조화 감상 메모로 저장하는 개인 감상 아카이브입니다.

이 저장소는 프로젝트 제출용 코드 저장소입니다. 로컬 DB, 업로드한 음성 파일, 모델 weight/cache, 결과보고서 PDF, 기획서 원본은 포함하지 않습니다.

## Repository Contents

- `app.py`: Streamlit 애플리케이션 진입점과 화면 렌더링
- `src/`: DB, LLM, 로컬 AI, 프롬프트, 질문, 작품 맥락 조회 로직
- `scripts/`: 데모 데이터 생성과 임베딩 색인 재생성 스크립트
- `tests/`: MVP 흐름, DB, 검색, Wikipedia 맥락 조회 테스트
- `.env.example`: 로컬 AI 실행 설정 예시
- `.streamlit/config.toml`: Streamlit UI 테마와 실행 설정

## Current Milestone

현재 구현 범위:

- Streamlit 기반 로컬 우선 MVP
- 작품 단위 아카이브: 작품마다 감상 글 1개 유지
- 이어 기록 시 새 기록을 추가하지 않고 기존 감상 글 업데이트
- 기존 작품 감지 후 `기존 글 업데이트` / `새 작품으로 만들기` 분리
- 텍스트/음성 기록 모드 선택
- 기본 질문 5개와 선택형 파생 질문
- AI 구조화 메모 초안 생성, 저장 전 수정, 저장 후 수정
- SQLite 저장, 키워드 검색, 의미 검색, 연결된 감상 표시
- Wikipedia EN 기반 작품 맥락 자동 참고
- faster-whisper 기반 음성 파일 전사
- Qwen3-TTS 기반 질문 음성 출력
- Ollama / OpenAI / mock LLM provider 분리

이미지 생성과 30초 요약은 MVP 범위에서 제외했습니다.

## Setup

```powershell
conda activate scrapbook-ai
pip install -r requirements.txt
```

로컬 AI 기능까지 사용할 경우:

```powershell
conda activate scrapbook-ai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-local-ai.txt
winget install Ollama.Ollama
ollama pull qwen3.5:9b
```

모델 파일은 저장소에 포함하지 않고, Ollama 저장소와 Hugging Face cache에서 로컬로 관리합니다.

## Run

```powershell
conda activate scrapbook-ai
streamlit run app.py
```

## Test

```powershell
conda activate scrapbook-ai
python -m compileall app.py src tests scripts
python -m unittest discover -s tests
```

## Environment

`.env.example`을 `.env`로 복사한 뒤 필요한 provider만 조정합니다.

기본 로컬 설정:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_THINK=false
OLLAMA_TIMEOUT_SECONDS=600
OLLAMA_NUM_PREDICT=3500

STT_MODEL=Systran/faster-whisper-large-v3
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

WIKIPEDIA_CONTEXT_ENABLED=true
WIKIPEDIA_LANGUAGE=en
WIKIPEDIA_PROMPT_MAX_CHARS=7000
```

`OLLAMA_THINK=true`로 바꾸면 Qwen thinking을 켤 수 있습니다. 다만 JSON 응답 안정성을 위해 기본값은 `false`입니다.

제목 입력 후 앱은 Wikipedia EN에서 작품 정보를 자동으로 가져와 질문 생성과 감상 메모 작성의 보조 맥락으로 사용합니다. 사용자 감상이 본문이며, Wikipedia 정보는 작품 식별과 배경 이해에만 사용하도록 프롬프트에서 제한합니다. 네트워크 실패 시 기존 기록 흐름은 계속 동작합니다.

`.env`에는 개인 API key가 들어갈 수 있으므로 Git에 올리지 않습니다.

## Data Model

핵심 구조:

```text
work
ㄴ record
```

- `scrapbook_works`: 작품 제목, 유형, 현재 감상 상태
- `scrapbook_records`: 작품별 대표 감상 글, 입력 모드, 감상 단계, 질문 답변, 생성/수정 메모, 외부 작품 맥락 JSON
- `record_embeddings`: 의미 검색과 연결된 감상 추천용 벡터
- `record_assets`: 입력 음성, 질문 TTS 등 로컬 파일 자산

기존 단일 기록 데이터는 앱 실행 시 자동으로 작품 단위로 묶입니다. 같은 작품으로 이어 기록하면 대표 감상 글이 업데이트됩니다.

## Local AI Features

- LLM: Ollama `qwen3.5:9b`
- Work context: Wikipedia EN Action API
- STT: `Systran/faster-whisper-large-v3`
- Embedding: `Qwen/Qwen3-Embedding-0.6B`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Image generation: excluded

의미 검색 색인은 저장/수정, 의미 검색, 연결된 감상 조회 때 자동 보정됩니다. 필요하면 sidebar의 `관리용 색인`에서 수동으로 다시 만들 수 있고, 아래 명령도 사용할 수 있습니다.

```powershell
python scripts/rebuild_embedding_index.py
```

## Product Scope

1. 작품 생성 및 기존 작품 감지
2. 같은 작품의 기존 감상 글 업데이트
3. 텍스트/음성 기반 감상 인터뷰
4. AI 구조화 메모 생성과 수정
5. 작품별 감상 글 상세, 검색, 연결된 감상
6. Wikipedia 기반 작품 맥락 보조
