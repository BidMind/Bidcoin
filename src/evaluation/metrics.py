"""
metrics.py — RAG 평가 지표 계산 모듈

지원 지표:
  [검색(Retrieval) — LLM 불필요]
    - hit_rate          : 상위 K개 결과에 정답 문서가 1개 이상 포함된 비율
    - precision_at_k    : 상위 K개 결과 중 정답 문서의 비율
    - recall_at_k       : 전체 정답 문서 중 상위 K개에 포함된 비율
    - mrr               : Mean Reciprocal Rank (첫 번째 정답의 순위 역수 평균)
    - ndcg_at_k         : Normalized Discounted Cumulative Gain

  [생성(Generation) — LLM 불필요]
    - token_overlap_f1  : 생성 답변과 참조 답변 간 토큰 F1 (어휘 기반)
    - answer_in_context : 참조 정답이 검색된 컨텍스트 내에 포함되는지 여부 (Faithfulness 근사)

  [LLM-as-Judge — LLM 필요]
    - llm_judge_evaluate : LLM이 직접 채점하는 6개 지표 동시 계산
                           relevance        (answer_relevancy 대응)
                           faithfulness     (faithfulness 대응)
                           correctness      (answer_correctness 대응)
                           completeness     (answer_correctness 보완)
                           context_precision(context_precision 대응)
                           context_recall   (context_recall 대응)
"""

from __future__ import annotations

import json
import math
import re
import warnings
from typing import List, Optional


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """한국어/영어 혼용 텍스트를 공백·특수문자 기준으로 토큰화."""
    return re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# 검색 지표
# ---------------------------------------------------------------------------

def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """상위 K 결과에 정답이 1개 이상 있으면 1.0, 없으면 0.0."""
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Precision@K = |retrieved ∩ relevant| / |retrieved|"""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Recall@K = |retrieved ∩ relevant| / |relevant|"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank (단일 쿼리). 첫 번째 정답 위치의 역수."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """NDCG@K. 관련 문서를 이진 relevance(0/1)로 계산."""
    relevant_set = set(relevant_ids)

    def dcg(ids: List[str]) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, rid in enumerate(ids, start=1)
            if rid in relevant_set
        )

    actual_dcg = dcg(retrieved_ids)
    ideal_ids = [rid for rid in retrieved_ids if rid in relevant_set] + \
                [rid for rid in retrieved_ids if rid not in relevant_set]
    ideal_dcg = dcg(ideal_ids)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 생성 지표
# ---------------------------------------------------------------------------

def token_overlap_f1(prediction: str, reference: str) -> float:
    """
    예측 답변과 참조 답변 간의 토큰 수준 F1 점수.
    간단한 어휘 기반 비교로, LLM 없이 빠르게 계산 가능.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    common = pred_set & ref_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    return 2 * precision * recall / (precision + recall)


def answer_in_context(reference_answer: str, context_chunks: List[str]) -> float:
    """
    참조 정답의 핵심 토큰이 검색된 컨텍스트 청크 안에 존재하는지 확인.
    Faithfulness(충실성)의 근사 지표 — LLM 호출 없이 계산.

    Returns:
        0.0 ~ 1.0: 참조 정답 토큰 중 컨텍스트에 등장하는 비율.
    """
    ref_tokens = set(_tokenize(reference_answer))
    if not ref_tokens:
        return 0.0

    all_context_tokens = set(_tokenize(" ".join(context_chunks)))
    overlap = ref_tokens & all_context_tokens
    return len(overlap) / len(ref_tokens)


# ---------------------------------------------------------------------------
# 집계 유틸
# ---------------------------------------------------------------------------

def aggregate_metrics(results: List[dict]) -> dict:
    """
    개별 쿼리 결과 리스트를 받아 평균 지표를 계산한다.

    Args:
        results: evaluator.py 가 생성하는 per-query result dict 리스트.
                 각 dict 는 'hit_rate', 'precision', 'recall', 'mrr',
                 'ndcg', 'f1', 'faithfulness' 키를 포함해야 한다.

    Returns:
        각 지표의 평균값을 담은 dict.
    """
    if not results:
        return {}

    keys = ["hit_rate", "precision", "recall", "mrr", "ndcg", "f1", "faithfulness"]
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results if key in r]
        aggregated[key] = sum(values) / len(values) if values else 0.0

    aggregated["n_queries"] = len(results)
    return aggregated


# ---------------------------------------------------------------------------
# LLM-as-Judge 지표 (LLM 필요)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "당신은 RAG(검색 증강 생성) 시스템의 답변 품질을 평가하는 전문가입니다. "
    "주어진 기준에 따라 1~5점 척도로 점수를 매기고, 반드시 JSON 형식으로만 응답하세요."
)

_JUDGE_USER_PROMPT = """\
다음 항목을 평가하세요.

**질문:** {question}

**생성된 답변:** {answer}

**참조 정답:** {reference}

**검색된 컨텍스트:**
{context}

아래 6가지 기준에 따라 각 항목을 1~5점으로 평가하세요.

[답변 품질]
- relevance        (관련성):         답변이 질문에 직접적으로 답하고 있는가?
- faithfulness     (충실성):         답변의 모든 주장이 검색된 컨텍스트에 근거하는가? 컨텍스트에 없는 내용을 지어냈다면 감점.
- correctness      (정확성):         답변이 참조 정답의 핵심 사실과 일치하는가?
- completeness     (완전성):         답변이 참조 정답의 모든 핵심 정보를 빠짐없이 포함하는가?

[검색 품질]
- context_precision(컨텍스트 정밀도): 검색된 컨텍스트 청크들이 질문 답변에 실제로 유용한가? 무관한 청크가 많을수록 감점.
- context_recall   (컨텍스트 재현율): 참조 정답을 도출하는 데 필요한 핵심 정보가 검색된 컨텍스트에 모두 있는가?

점수 기준: 1=매우나쁨, 2=나쁨, 3=보통, 4=좋음, 5=매우좋음

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{"relevance": <1-5>, "faithfulness": <1-5>, "correctness": <1-5>, "completeness": <1-5>, "context_precision": <1-5>, "context_recall": <1-5>, "reason": "<한 문장 이유>"}}\
"""

_ALL_CRITERIA = [
    "relevance", "faithfulness", "correctness", "completeness",
    "context_precision", "context_recall",
]


def _parse_judge_response(text: str) -> dict:
    """LLM 응답 텍스트에서 JSON을 파싱한다. 마크다운 코드블록도 처리."""
    # 마크다운 코드블록 제거
    text = re.sub(r"```(?:json)?", "", text).strip()
    # 첫 번째 { ... } 블록 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON을 찾을 수 없음: {text!r}")
    return json.loads(match.group())


def _call_llm(llm, system: str, user: str) -> str:
    """LangChain LLM 또는 OpenAI 클라이언트로 응답을 받아 문자열로 반환."""
    # LangChain interface (invoke)
    if hasattr(llm, "invoke"):
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # OpenAI client (openai>=1.0)
    if hasattr(llm, "chat"):
        response = llm.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    raise TypeError(f"지원하지 않는 LLM 타입: {type(llm)}")


def llm_judge_evaluate(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    references: List[str],
    llm=None,
    criteria: Optional[List[str]] = None,
) -> dict:
    """
    LLM-as-Judge 방식으로 생성 답변과 검색 품질을 평가한다.

    RAGAS 5개 지표에 대응하는 6개 기준을 단일 LLM 호출로 채점한다.
    어휘 기반 지표와 달리 한국어 구조화 답변에서도 안정적으로 동작한다.

    지표 대응표:
        relevance         ↔  RAGAS answer_relevancy
        faithfulness      ↔  RAGAS faithfulness
        correctness       ↔  RAGAS answer_correctness (핵심 사실 일치)
        completeness      ↔  RAGAS answer_correctness (누락 정보)
        context_precision ↔  RAGAS context_precision
        context_recall    ↔  RAGAS context_recall

    Args:
        questions  : 질문 문자열 리스트
        answers    : 생성된 답변 리스트
        contexts   : 각 질문에 대해 검색된 청크 리스트의 리스트
        references : 참조 정답 리스트
        llm        : LangChain LLM 객체 또는 None.
                     None 이면 OPENAI_API_KEY 환경변수로 gpt-4o-mini 자동 사용.
                     예) from langchain_openai import ChatOpenAI
                         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                     예) from langchain_anthropic import ChatAnthropic
                         llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
        criteria   : 집계에 포함할 기준 리스트. None 이면 6개 전체.
                     ["relevance", "faithfulness", "correctness", "completeness",
                      "context_precision", "context_recall"]

    Returns:
        {
            "relevance":         float,   # 0.0 ~ 1.0 (1~5점 → /5 정규화)
            "faithfulness":      float,
            "correctness":       float,
            "completeness":      float,
            "context_precision": float,
            "context_recall":    float,
            "judge_score":       float,   # criteria 지표들의 평균
            "per_query":         List[dict]  # 질문별 원시 점수 및 이유
        }

    사용 예시:
        import os
        os.environ["OPENAI_API_KEY"] = "sk-..."

        result = llm_judge_evaluate(
            questions=["봉화군 재난통합관리시스템 예산은?"],
            answers=["9억 원"],
            contexts=[["봉화군 재난통합관리시스템 고도화 사업 예산: 900,000,000원 ..."]],
            references=["900,000,000원"],
        )
        # {"relevance": 1.0, "faithfulness": 0.9, "correctness": 0.9,
        #  "context_precision": 1.0, "context_recall": 1.0, ...}
    """
    if criteria is None:
        criteria = _ALL_CRITERIA

    # llm 이 None 이면 OpenAI gpt-5-mini 자동 생성
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        except ImportError:
            raise ImportError(
                "llm=None 으로 사용하려면 langchain-openai 와 OPENAI_API_KEY 가 필요합니다.\n"
                "  pip install langchain-openai\n"
                "  export OPENAI_API_KEY=sk-..."
            )

    per_query = []
    accumulated: dict = {c: [] for c in criteria}

    for q, a, ctx, ref in zip(questions, answers, contexts, references):
        ctx_text = "\n---\n".join(ctx) if ctx else "(컨텍스트 없음)"
        user_msg = _JUDGE_USER_PROMPT.format(
            question=q,
            answer=a,
            reference=ref,
            context=ctx_text,
        )
        try:
            raw = _call_llm(llm, _JUDGE_SYSTEM_PROMPT, user_msg)
            parsed = _parse_judge_response(raw)

            row = {"question": q, "answer": a}
            for c in _ALL_CRITERIA:
                raw_score = parsed.get(c)
                # 1~5 점수를 0~1로 정규화
                score = (float(raw_score) - 1) / 4 if raw_score is not None else None
                row[c] = score
                if c in criteria and score is not None:
                    accumulated[c].append(score)
            row["reason"] = parsed.get("reason", "")
            row["error"] = None
        except Exception as e:
            row = {
                "question": q, "answer": a,
                "relevance": None, "faithfulness": None,
                "correctness": None, "completeness": None,
                "reason": "", "error": str(e),
            }

        per_query.append(row)

    # 집계
    output: dict = {}
    for c in _ALL_CRITERIA:
        vals = accumulated.get(c, [])
        output[c] = sum(vals) / len(vals) if vals else None

    valid = [output[c] for c in criteria if output.get(c) is not None]
    output["judge_score"] = sum(valid) / len(valid) if valid else None
    output["per_query"] = per_query

    return output
