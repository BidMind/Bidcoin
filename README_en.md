# Bid Mind — Korean B2G RFP RAG System

Advanced RAG pipeline for Korean public procurement (B2G) RFP documents.  
Answers questions about bid opportunities from HWP/PDF proposal documents using hybrid retrieval and LLM-as-Judge evaluation.

---

## Features

- **Hybrid Search** — FAISS (dense/semantic) + BM25 (sparse/keyword) merged with Reciprocal Rank Fusion (RRF)
- **Multi-Query Reformulation** — decomposes complex questions into targeted sub-queries with format-aware routing
- **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 scores candidate chunks before generation
- **Self-RAG Guard** — skips LLM fact-check when reranker confidence ≥ 0.5; evaluates otherwise
- **16 Response Formats** — fact, condition, summary, compare, recommend_score, complex_strategy, follow_up, etc.
- **LLM-as-Judge Evaluation** — 6 RAGAS-compatible metrics: relevance, faithfulness, correctness, completeness, context_precision, context_recall

---

## Architecture

```
User Query
    │
    ▼
[Router]  ──── CHITCHAT ────► direct reply
    │ RAG
    ▼
[Reformulator]
    │  queries[], filters{}, format_hint
    ▼
[Multi-Query Hybrid Search]  (for each query)
    ├── FAISS (OpenAI text-embedding-3-small)
    └── BM25  (Kiwi morphological tokenizer)
         │
         ▼ RRF merge → dedup
[Cross-Encoder Reranker]  (BAAI/bge-reranker-v2-m3)
    │
    ▼ score filter (≥ 0.25)
[Self-RAG Evaluator]
    │  pass-through if max_score ≥ 0.5
    ▼
[Context Builder]
    │
    ▼
[Generator]  (GPT-4o / format_hint → prompts.py)
    │
    ▼
Answer + Source Citations
```

---

## Directory Structure

```
Bidcoin/
├── rag_api_v4.py           # Main RAG pipeline entry point
├── eval.py                 # Evaluation runner
├── config.py               # Env vars and paths
├── app/
│   ├── streamlit_app.py    # Streamlit chat UI
│   └── convert.py          # HWP→PDF converter for UI
├── scripts/
│   ├── build_index.py      # Build FAISS + BM25 indexes
│   └── run_evaluation.py   # Batch evaluation runner
├── src/
│   ├── parsing/
│   │   ├── hwp_parser_v2.py    # HWP binary → text (olefile + zlib + bs4)
│   │   ├── pdf_parser.py       # PDF → text (pymupdf)
│   │   └── concat.py           # Full ingestion pipeline
│   ├── ingestion/
│   │   └── chunker_v2.py       # Section/table/text chunker with meta prefix
│   ├── preprocessing/
│   │   └── metadata_cleaning.py
│   ├── embedding/
│   │   ├── embedder.py         # OpenAI embedding wrapper
│   │   └── vector_store_v2.py  # Build/load FAISS + BM25 indexes
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid FAISS+BM25+RRF search
│   │   └── reranker.py         # Cross-Encoder reranker
│   ├── modules/
│   │   ├── router.py           # Semantic router (RAG vs CHITCHAT)
│   │   ├── reformulator.py     # Multi-query decomposition + format_hint
│   │   ├── evaluator.py        # Self-RAG evaluator
│   │   ├── compressor.py       # Context compressor (optional)
│   │   └── hyde.py             # HyDE (disabled in v4)
│   ├── generation/
│   │   ├── prompts.py          # 16 response format templates
│   │   ├── generator.py        # LLM generation orchestrator
│   │   ├── context_builder.py  # Context block formatter
│   │   ├── llm.py              # OpenAI client wrapper
│   │   └── schemas.py          # Pydantic data models
│   └── evaluation/
│       ├── metrics.py          # LLM-as-Judge 6 metrics
│       ├── evaluator.py        # Batch evaluation logic
│       └── question_sets.py    # Eval question bank (47 questions, 12 categories)
├── faiss_index/
│   ├── index.faiss
│   ├── index.pkl
│   └── bm25.pkl
├── src/results/                # Evaluation JSON outputs
└── notebooks/                  # Exploratory notebooks
```

---

## Setup

**Requirements:** Python 3.12, CUDA optional (CPU fallback supported)

```bash
# 1. Clone and enter project
git clone <repo-url> && cd Bidcoin

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip<26" "setuptools==81.0.0" wheel
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — required:
#   OPENAI_API_KEY=sk-...
#   DATABASE_DIR=./data          # folder containing HWP/PDF source documents
#   OUTPUT_DIR=./output

# 4. Build indexes (run once after adding documents)
python update_entire_pipeline.py
```

> **HWP parsing** requires no external tools — the parser uses `olefile` + `zlib` + `beautifulsoup4` directly.  
> For documents that fail binary parsing, install `libreoffice` as a fallback converter.

---

## Usage

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

### Run Evaluation

```bash
python eval.py
# Results saved to Bidcoin/eval_results_<timestamp>.json
```

---

## Evaluation Results

Latest results across 47 questions / 12 categories (best run: `20260420_072555.json`):

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

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Removed HyDE in v4 | Cost and latency savings outweighed the marginal recall gain |
| fact format = exactly 1 query | Multiple queries add noise that degrades context_precision for single-fact lookups |
| Date field names excluded from queries | "공개 일자" never appears as a document title — causes zero retrieval |
| Abbreviations get dual queries | e.g. UICC → ["기관명 UICC 사업명", "기관명 업무키워드 사업"] |
| BM25 tokenized with Kiwi | Raw space-tokenization misses Korean compound nouns critical for procurement terminology |
| Self-RAG pass-through at ≥ 0.5 | Avoids LLM evaluation overhead when the reranker is already highly confident |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `DATABASE_DIR` | Yes | `./data` | Source HWP/PDF document folder |
| `OUTPUT_DIR` | No | `./output` | Processed data output folder |
| `EMBED_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `RERANK_MODEL` | No | `BAAI/bge-reranker-v2-m3` | HuggingFace reranker model ID |
| `CLAUDE_API_KEY` | No | — | Reserved for Claude migration |
