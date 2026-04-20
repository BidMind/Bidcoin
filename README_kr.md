# Bid Mind — 한국 공공조달 RFP RAG 시스템

한국 B2G(기업 대 정부) 공공 조달 입찰 문서를 위한 고급 RAG 파이프라인.  
HWP/PDF 제안요청서 문서를 기반으로 입찰 기회에 관한 질문에 하이브리드 검색 및 LLM 평가 방식으로 답변합니다.

---

## 주요 기능

- **하이브리드 검색** — FAISS(밀집 벡터/의미 검색) + BM25(희소/키워드 검색)를 RRF(Reciprocal Rank Fusion)로 병합
- **다중 쿼리 재구성** — 복잡한 질문을 형식 인식 라우팅과 함께 세분화된 하위 쿼리로 분해
- **Cross-Encoder 재정렬** — BAAI/bge-reranker-v2-m3 모델로 후보 청크를 생성 전 점수화
- **Self-RAG 가드** — 재정렬 신뢰도 ≥ 0.5이면 LLM 팩트 검증을 생략하고, 미달 시 자가 평가 진행
- **16가지 응답 형식** — fact, condition, summary, compare, recommend_score, complex_strategy, follow_up 등
- **LLM-as-Judge 평가** — 6가지 RAGAS 호환 지표: 관련성, 충실성, 정확성, 완전성, 컨텍스트 정밀도, 컨텍스트 재현율

---

## 시스템 구조

```
사용자 질문
    │
    ▼
[라우터]  ──── 일상 대화 ────► 직접 응답
    │ RAG
    ▼
[재구성기 (Reformulator)]
    │  queries[], filters{}, format_hint
    ▼
[다중 쿼리 하이브리드 검색]  (쿼리별 실행)
    ├── FAISS (OpenAI text-embedding-3-small)
    └── BM25  (Kiwi 형태소 분석기 토크나이저)
         │
         ▼ RRF 병합 → 중복 제거
[Cross-Encoder 재정렬]  (BAAI/bge-reranker-v2-m3)
    │
    ▼ 점수 필터 (≥ 0.25)
[Self-RAG 평가기]
    │  최고 점수 ≥ 0.5이면 프리패스
    ▼
[컨텍스트 빌더]
    │
    ▼
[생성기]  (GPT-4o / format_hint → prompts.py)
    │
    ▼
답변 + 출처 인용
```

---

## 디렉토리 구조

```
Bidcoin/
├── rag_api_v4.py           # 메인 RAG 파이프라인 진입점
├── eval.py                 # 평가 실행기
├── config.py               # 환경 변수 및 경로 설정
├── app/
│   ├── streamlit_app.py    # Streamlit 채팅 UI
│   └── convert.py          # UI용 HWP→PDF 변환기
├── scripts/
│   ├── build_index.py      # FAISS + BM25 인덱스 빌드
│   └── run_evaluation.py   # 배치 평가 실행기
├── src/
│   ├── parsing/
│   │   ├── hwp_parser_v2.py    # HWP 바이너리 → 텍스트 (olefile + zlib + bs4)
│   │   ├── pdf_parser.py       # PDF → 텍스트 (pymupdf)
│   │   └── concat.py           # 전체 수집 파이프라인
│   ├── ingestion/
│   │   └── chunker_v2.py       # 섹션/표/텍스트 청커 (메타 접두사 포함)
│   ├── preprocessing/
│   │   └── metadata_cleaning.py
│   ├── embedding/
│   │   ├── embedder.py         # OpenAI 임베딩 래퍼
│   │   └── vector_store_v2.py  # FAISS + BM25 인덱스 빌드/로드
│   ├── retrieval/
│   │   ├── retriever.py        # 하이브리드 FAISS+BM25+RRF 검색
│   │   └── reranker.py         # Cross-Encoder 재정렬기
│   ├── modules/
│   │   ├── router.py           # 의미론적 라우터 (RAG vs 일상 대화)
│   │   ├── reformulator.py     # 다중 쿼리 분해 + format_hint
│   │   ├── evaluator.py        # Self-RAG 평가기
│   │   ├── compressor.py       # 컨텍스트 압축기 (선택적)
│   │   └── hyde.py             # HyDE (v4에서 비활성화)
│   ├── generation/
│   │   ├── prompts.py          # 16가지 응답 형식 템플릿
│   │   ├── generator.py        # LLM 생성 오케스트레이터
│   │   ├── context_builder.py  # 컨텍스트 블록 포매터
│   │   ├── llm.py              # OpenAI 클라이언트 래퍼
│   │   └── schemas.py          # Pydantic 데이터 모델
│   └── evaluation/
│       ├── metrics.py          # LLM-as-Judge 6가지 지표
│       ├── evaluator.py        # 배치 평가 로직
│       └── question_sets.py    # 평가 질문 세트 (47개 질문, 12개 카테고리)
├── faiss_index/
│   ├── index.faiss
│   ├── index.pkl
│   └── bm25.pkl
├── src/results/                # 평가 JSON 출력 결과
└── notebooks/                  # 탐색용 노트북
```

---

## 설치

**요구사항:** Python 3.12, CUDA 선택적 (CPU 폴백 지원)

```bash
# 1. 클론 및 디렉토리 이동
git clone <repo-url> && cd Bidcoin

# 2. 가상환경 생성
python -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip<26" "setuptools==81.0.0" wheel
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일 수정 — 필수 항목:
#   OPENAI_API_KEY=sk-...
#   DATABASE_DIR=./data          # HWP/PDF 원본 문서 폴더
#   OUTPUT_DIR=./output

# 4. 인덱스 빌드 (문서 추가 후 최초 1회 실행)
python update_entire_pipeline.py
```

> **HWP 파싱**은 외부 도구 없이 `olefile` + `zlib` + `beautifulsoup4`를 직접 사용합니다.  
> 바이너리 파싱이 실패하는 문서는 `libreoffice`를 폴백 변환기로 설치하세요.

---

## 사용법

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### Python API

```python
from rag_api_v4 import get_rag_context
from src.generation.generator import BidCoinGenerator

result = get_rag_context("고려대학교 차세대 포털 사업의 예산은?", chat_history=[])
# result = {"contexts": [...], "format_hint": "fact", ...}

generator = BidCoinGenerator()
answer = generator.generate(result)
```

### CLI

```bash
python cli.py
```

### 평가 실행

```bash
python eval.py
# 결과는 Bidcoin/eval_results_<타임스탬프>.json 에 저장됩니다
```

---

## 평가 결과

47개 질문 / 12개 카테고리 최신 결과 (최고 실행: `20260420_072555`):

	relevance	faithfulness	correctness	completeness	context_precision	context_recall
category						
금액	1.000000	1.00	0.972222	0.972222	0.861111	1.000000
기간	1.000000	1.00	1.000000	1.000000	0.750000	1.000000
기관	1.000000	1.00	1.000000	1.000000	0.850000	1.000000
다문서추천	0.700000	0.85	0.350000	0.250000	0.750000	0.300000
복합	1.000000	1.00	1.000000	1.000000	0.875000	1.000000
비교	0.916667	1.00	0.666667	0.416667	0.750000	0.333333
사업내용	1.000000	1.00	1.000000	1.000000	0.750000	1.000000
인사이트	0.535714	1.00	0.285714	0.250000	0.464286	0.178571
종합	0.833333	1.00	1.000000	0.583333	0.833333	0.750000
추천	0.750000	0.75	0.583333	0.500000	0.666667	0.666667
확인불가	1.000000	1.00	1.000000	1.000000	0.666667	0.666667
후속	1.000000	1.00	1.000000	0.916667	0.833333	1.000000

---

## 주요 설계 결정

| 결정 | 이유 |
|---|---|
| v4에서 HyDE 제거 | 비용 및 지연 절감 효과가 재현율 향상보다 컸음 |
| fact 형식 = 정확히 1개 쿼리 | 다중 쿼리가 단일 사실 조회 시 컨텍스트 정밀도를 낮추는 노이즈를 발생시킴 |
| 날짜 필드명을 쿼리에서 배제 | "공개 일자"는 문서 제목에 등장하지 않아 검색 결과 0건을 유발 |
| 약어는 이중 쿼리 생성 | 예: UICC → ["기관명 UICC 사업명", "기관명 업무키워드 사업"] |
| BM25 토큰화에 Kiwi 사용 | 공백 분리만으로는 조달 용어의 한국어 복합명사를 처리하지 못함 |
| Self-RAG 프리패스 임계값 ≥ 0.5 | 재정렬기 신뢰도가 이미 높을 때 LLM 평가 오버헤드를 회피 |

---

## 환경 변수

| 변수명 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|
| `OPENAI_API_KEY` | 필수 | — | OpenAI API 키 |
| `DATABASE_DIR` | 필수 | `./data` | HWP/PDF 원본 문서 폴더 |
| `OUTPUT_DIR` | 선택 | `./output` | 처리된 데이터 출력 폴더 |
| `EMBED_MODEL` | 선택 | `text-embedding-3-small` | OpenAI 임베딩 모델 |
| `RERANK_MODEL` | 선택 | `BAAI/bge-reranker-v2-m3` | HuggingFace 재정렬 모델 ID |
| `CLAUDE_API_KEY` | 선택 | — | Claude 전환 예비용 |

## 보고서 파일
본문 프로젝트_최종보고서.pdf 참고

## 협업일지
https://www.notion.so/Daily-4-0870b7469455830c888901126c96b2eb?source=copy_link
