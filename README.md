# review-scrapbook-ai

문화콘텐츠 감상 직후의 짧은 단상을 AI 질문으로 확장하고, 나중에 다시 꺼내볼 수 있는 구조화된 감상 스크랩으로 저장하는 프로젝트입니다.

## Current Milestone

현재 구현된 1차 MVP:

- Streamlit 기반 텍스트 입력 UI
- 작품 종류, 제목, 감상 상태, 짧은 감상 입력
- 후속 질문 3~5개 생성
- 답변 기반 구조화 메모 생성
- SQLite 저장 및 아카이브 조회
- `mock`, `openai`, `ollama` LLM provider 분리

기본값은 `mock` provider라 API 키 없이 바로 실행할 수 있습니다.

## Setup

```powershell
conda activate scrapbook-ai
pip install -r requirements.txt
```

## Run

```powershell
conda activate scrapbook-ai
streamlit run app.py
```

## Optional LLM Providers

기본 provider는 `mock`입니다. 실제 LLM을 붙일 때는 repo 루트에 `.env`를 만들고 아래 중 하나를 설정합니다.

OpenAI:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Ollama:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_URL=http://localhost:11434/api/generate
```

## Next Steps

1. 텍스트 MVP를 샘플 감상으로 검증한다.
2. 실제 LLM provider를 연결하고 프롬프트 품질을 조정한다.
3. `faster-whisper`로 음성 파일 업로드 기반 STT를 붙인다.
4. 임베딩 검색으로 유사 감상/같은 작품 복기 기능을 추가한다.
