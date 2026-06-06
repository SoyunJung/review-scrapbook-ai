# review-scrapbook-ai

감상 직후 떠오른 짧은 생각을 질문으로 정리하고, 다시 읽기 좋은 감상 메모로 저장하는 Streamlit 앱입니다.

## 주요 기능

* 작품별 감상 기록 저장
* 텍스트 / 음성 감상 인터뷰
* LLM 기반 파생 질문 생성
* AI 감상 메모 초안 생성
* 기존 작품 이어 기록
* 키워드 검색 / 의미 검색
* 유사한 감상 기록 추천
* SQLite 로컬 저장

## 사용한 모델

* LLM: Ollama `qwen3.5:9b`
* STT: `Systran/faster-whisper-large-v3`
* Embedding: `Qwen/Qwen3-Embedding-0.6B`
* TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
* 작품 맥락 참고: Wikipedia EN API

## 프로젝트 구조

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

## 설치

Windows / PowerShell 기준입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

로컬 AI 기능까지 실행하려면 추가 패키지를 설치합니다.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-local-ai.txt
```

```powershell
winget install Ollama.Ollama
ollama pull qwen3.5:9b
```

## 환경 설정

```powershell
Copy-Item .env.example .env
```

## 실행

```powershell
streamlit run app.py
```


## 기본 흐름

1. 작품 정보 입력
2. 텍스트 또는 음성으로 질문에 답변
3. AI가 감상 메모 초안 생성
4. 사용자가 수정 후 저장
5. 검색 또는 연결된 감상으로 다시 확인

## 참고

* 이미지 생성 기능은 최종 구현에서 제외했습니다.
* 의미 검색 색인을 다시 만들려면 아래 명령을 사용합니다.

```powershell
python scripts/rebuild_embedding_index.py
```
