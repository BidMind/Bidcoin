# Generation README

## 1. 이 문서의 목적

이 문서는 `generation` 내의 코드가 어떤 흐름으로 작동하는지, 각 파일이 어떤 역할을 하는지, 그리고 Retrieval 결과가 들어온 뒤 Generation이 어떻게 답변을 만드는지를 빠르게 이해할 수 있도록 정리한 문서입니다.

이 코드는 Generation 담당용 베이스라인이기에 원본 PDF/HWP를 직접 읽어 검색하는 역할은 하지 않습니다.

대신, Retrieval 단계가 넘겨준 결과를 입력으로 받아 프롬프트를 만들고 LLM을 호출해 최종 답변을 생성합니다.

---

## 2. 요약 및 흐름

이 프로젝트의 Generation 코드는 아래 흐름으로 작동합니다.

```text
사용자 질문
→ Retrieval 결과(question, contexts, chat_history) 입력
→ context_builder.py가 문서 조각을 LLM이 읽기 좋은 문자열로 정리
→ prompts.py가 system prompt + user prompt 생성
→ llm.py가 OpenAI API 호출
→ generator.py가 전체 과정을 묶어 최종 답변 반환
→ cli.py / notebook 에서 실행 및 확인
```

---

## 3. 입력과 출력

### 입력

Generation이 기대하는 입력은 `RetrievalResult` 구조입니다.

```json
{
  "question": "콘텐츠 관리 요구사항과 보안 요구사항을 정리해줘.",
  "contexts": [
    {
      "chunk_id": "doc_001_chunk_01",
      "text": "청크 본문",
      "source_file": "국민연금공단_이러닝시스템.hwp",
      "organization": "국민연금공단",
      "project_name": "이러닝시스템 구축",
      "summary": "문서 전체 요약",
      "score": 0.93
    }
  ],
  "chat_history": [
    { "role": "user", "content": "국민연금공단 사업 문서를 찾아줘." },
    {
      "role": "assistant",
      "content": "국민연금공단 이러닝시스템 구축 관련 문서를 참고하겠습니다."
    }
  ]
}
```

### 출력

Generation의 최종 출력은 `GenerationResponse` 구조입니다.

- `answer`: 최종 답변
- `used_context_count`: 실제 사용한 context 개수
- `used_sources`: 출처 파일명 목록
- `context_preview`: 모델에 넣은 context 앞부분 미리보기
- `raw_model_output`: 모델 원문 출력

---

## 4. 실행 흐름 상세

### 4-1. 가장 바깥 실행 진입점

가장 간단한 실행은 `cli.py`입니다.

```bash
python -m src.bidcoin.cli
```

이 명령이 실행되면 내부적으로 다음이 일어납니다.

1. `mock_data.py`에서 가짜 Retrieval 결과 생성
2. `BidCoinGenerator()` 객체 생성
3. `generator.generate(retrieval_result)` 호출
4. answer / used_sources 출력

### 4-2. 실제 내부 함수 호출 순서

```text
cli.py
└─ get_mock_retrieval_result()
└─ BidCoinGenerator()
   └─ OpenAIClient(Settings)
└─ generator.generate(retrieval_result)
   ├─ build_context_block(contexts)
   ├─ build_history_block(chat_history)
   ├─ build_user_prompt(question, context_block, history_block)
   ├─ llm.generate_text(SYSTEM_PROMPT, user_prompt)
   └─ GenerationResponse 반환
```

---

## 5. 파일별 역할 설명

### 5-1. `requirements.txt`

필요한 라이브러리 목록입니다.

- `openai`: OpenAI API 호출
- `python-dotenv`: `.env` 파일 읽기
- `pydantic`: 입력/출력 스키마 검증
- `jupyter`, `ipykernel`: notebook 실행

---

### 5-2. `.env`

환경변수 필수 내용입니다. 꼭 `.env` 파일이 필요합니다.
이 파일은 git에 올라가지 않으므로 꼭 따로 만들어주세요.

```env
OPENAI_API_KEY=저희가 제공받은 키 넣으시면 됩니다.
```

---

### 5-3. `config.py`

설정 관리 파일입니다.

역할:

- `.env`를 읽음
- API 키 로드
- `validate()`로 설정이 유효한지 검사

---

### 5-4. `schemas.py`

Generation에서 사용하는 입력/출력 데이터 구조를 정의합니다.

#### 주요 클래스

- `ChatTurn`: 대화 한 턴
- `RetrievedContext`: Retrieval이 넘겨준 문서 조각 1개
- `RetrievalResult`: 질문 + context 리스트 + history 묶음
- `GenerationResponse`: Generation 결과 묶음

---

### 5-5. `prompts.py`

LLM에 전달할 프롬프트를 정의합니다.
가장 많이 손봐야 하는 모듈입니다.

#### `SYSTEM_PROMPT`

모델의 역할과 절대 규칙을 정합니다.

예:

- 문서에 없는 내용은 추측 금지
- 한국어로 답변
- 구조화된 형식
- 마지막에 출처 필수

#### `build_user_prompt(question, context_block, history_block)`

실제 질문마다 바뀌는 user prompt를 만듭니다.

입력:

- 질문
- 참고 문서 block
- 최근 대화 block

출력:

- 모델에게 그대로 넘길 최종 user prompt 문자열

=>

- `SYSTEM_PROMPT` : 고정 규칙이자 모델 성격을 정의한 것
- `build_user_prompt()` : 질문별 동적 입력

---

### 5-6. `mock_data.py`

테스트를 위한 가짜 데이터입니다.

역할:

- `RetrievalResult` 형식의 mock 예시 생성
- 질문, contexts, chat_history를 함께 넣어서 전체 흐름 테스트 가능하게 만듦

이 파일을 활용하여 Retrieval단계에서 넘겨주는 데이터가 없이도 아래가 가능합니다.

- prompt 실험
- answer format 테스트
- notebook 시연
- CLI 데모

---

### 5-7. `context_builder.py`

이 파일은 Retrieval이 넘긴 구조화 데이터를 LLM이 읽기 쉬운 문자열로 바꾸는 역할을 합니다.

#### 왜 필요한가?

Retrieval 결과는 보통 JSON/객체 형태입니다.
하지만 모델은 결국 문자열을 읽습니다.
따라서 context를 보기 좋게 정리해서 넣어야 합니다.

#### 주요 함수

##### `build_history_block(chat_history, max_turns=3)`

최근 대화 N턴을 문자열로 만듭니다.

예:

```text
- 사용자: 국민연금공단 사업 문서를 찾아줘.
- 어시스턴트: 국민연금공단 이러닝시스템 구축 관련 문서를 참고하겠습니다.
```

##### `_context_header(context, idx)`

각 문서 조각 앞에 붙는 헤더를 만듭니다.

예:

```text
[문서 1] chunk_id=doc_001_chunk_01 | score=0.9300 | 기관=국민연금공단 | 사업명=이러닝시스템 구축 | 파일명=국민연금공단_이러닝시스템.hwp
```

##### `build_context_block(contexts, max_contexts=3, max_chars=12000)`

context 리스트 전체를 하나의 긴 문자열로 조립합니다.

예:

```text
[문서 1] ...
문서 요약: ...
본문:
...

---

[문서 2] ...
문서 요약: ...
본문:
...
```

추가 역할:

- context 개수 제한
- 전체 글자 수 제한
- 출처 파일명 dedup

#### 왜 길이 제한이 필요한가?

- 비용 절감
- 불필요한 긴 입력 방지
- 관련도가 낮은 문서가 뒤에 너무 많이 붙는 것을 방지

#### 왜 `used_sources`를 따로 뽑는가?

최종 답변에 출처를 붙이거나, 디버깅할 때 어떤 파일을 썼는지 확인하기 위해서입니다.

---

### 5-8. `llm.py`

OpenAI API를 실제로 호출하는 파일입니다.

#### `OpenAIClient`

- `Settings`를 받아 검증
- OpenAI client 생성

#### `generate_text(instructions, user_input)`

- `instructions` = system prompt
- `user_input` = user prompt

를 받아 `responses.create()`를 호출합니다.

현재는 `response.output_text`만 반환하도록 하였습니다.
이유는 baseline 단계에서 최종 답변 텍스트만 있으면 충분하기 때문입니다.

---

### 5-9. `generator.py`

전체 Generation 오케스트레이션의 중심입니다. 거의 전체 흐름이 담겨 있습니다.

#### `BidCoinGenerator`

이 클래스가 실제 Generation 흐름을 관리합니다.

#### `generate(retrieval_result)` 내부 순서

1. `build_context_block()` 호출
2. `build_history_block()` 호출
3. `build_user_prompt()` 호출
4. `llm.generate_text()` 호출
5. `GenerationResponse`로 결과 반환

---

### 5-10. `cli.py`

가장 빠르게 결과를 확인하는 터미널 실행 파일입니다.

역할:

- mock retrieval 결과 생성
- generator 실행
- 질문 / 출처 / 답변 출력

---

### 5-11. `notebooks/01_generation_playground.ipynb`

실험용 notebook입니다.

보통 아래 순서로 사용합니다.

1. mock retrieval 결과 확인
2. context block 확인
3. history block 확인
4. SYSTEM_PROMPT / USER_PROMPT 확인
5. 실제 generation 호출
6. 질문을 바꿔가며 실험

여기서 실험할 수 있습니다.

---

## 6. 모델에 실제로 들어가는 입력은 어떻게 생기나?

### 6-1. system prompt

`prompts.py`의 `SYSTEM_PROMPT`

역할:

- 모델 역할 지정
- 절대 규칙 지정

예:

- 문서에 없는 내용은 추측 금지
- 마지막에 출처 표시

### 6-2. user prompt

`build_user_prompt()`가 만드는 문자열

구성:

- 참고 문서
- 최근 대화
- 질문
- 응답 지침
- 권장 답변 형식

### 6-3. 실제 호출 방식

`generator.py` 안에서 이렇게 연결됩니다.

```python
answer = self.llm.generate_text(
    instructions=SYSTEM_PROMPT,
    user_input=user_prompt,
)
```

`instructions` 에 system prompt이 들어가고, `user_input` 에 build_user_prompt 결과가 들어갑니다.

---

## 7. 현재 상태

### 구현된 것

- schema 정의
- context 조립
- history 조립
- prompt 생성
- OpenAI 호출
- generator orchestration
- CLI 실행
- notebook 실험
- mock data 테스트

### 아직 남은 것

- 실제 Retrieval 결과와 키 이름 맞추기
- 질문 유형별 프롬프트 고도화
- 모델 비교 실험
- history 전략 실험
- 평가 질문 세트 구성
- 실제 end-to-end 테스트

---
