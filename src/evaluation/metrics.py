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

  [RAGAS — LLM 필요]
    - ragas_evaluate    : RAGAS 공식 라이브러리로 5개 지표 동시 계산
                          faithfulness / answer_relevancy / context_precision
                          context_recall / answer_correctness
"""

from __future__ import annotations

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
# RAGAS 지표 (LLM 필요)
# ---------------------------------------------------------------------------

def ragas_evaluate(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    references: List[str],
    llm=None,
    embeddings=None,
    metrics: Optional[List] = None,
) -> dict:
    """
    RAGAS 공식 라이브러리를 사용해 5개 지표를 계산한다.

    Args:
        questions  : 질문 문자열 리스트
        answers    : 생성된 답변 리스트 (질문과 1:1 대응)
        contexts   : 각 질문에 대해 검색된 청크 리스트의 리스트
                     예) [["청크1", "청크2"], ["청크3"], ...]
        references : 참조 정답(ground truth) 리스트
        llm        : RAGAS 가 사용할 LLM 객체.
                     None 이면 환경변수 OPENAI_API_KEY 로 gpt-4o-mini 자동 사용.
                     직접 전달 예시:
                       from langchain_openai import ChatOpenAI
                       llm = ChatOpenAI(model="gpt-4o-mini")
                     또는:
                       from langchain_anthropic import ChatAnthropic
                       llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
        embeddings : RAGAS 가 사용할 임베딩 객체 (answer_relevancy, answer_correctness 에 필요).
                     None 이면 OpenAI text-embedding-3-small 자동 사용.
                     직접 전달 예시:
                       from langchain_openai import OpenAIEmbeddings
                       embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        metrics    : 계산할 RAGAS Metric 객체 리스트.
                     None 이면 5개 전체(faithfulness, answer_relevancy,
                     context_precision, context_recall, answer_correctness) 사용.

    Returns:
        {"faithfulness": float, "answer_relevancy": float,
         "context_precision": float, "context_recall": float,
         "answer_correctness": float, "ragas_score": float}
        계산 실패 시 해당 키 값은 None.

    사용 예시:
        import os
        os.environ["OPENAI_API_KEY"] = "sk-..."

        result = ragas_evaluate(
            questions=["봉화군 재난통합관리시스템 예산은?"],
            answers=["9억 원"],
            contexts=[["[공고번호: 20240430896 | ...] 봉화군 재난통합관리시스템 고도화 사업..."]],
            references=["900,000,000원"],
        )
        print(result)
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")

        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics.collections import (
            faithfulness as _faithfulness,
            answer_relevancy as _answer_relevancy,
            context_precision as _context_precision,
            context_recall as _context_recall,
            answer_correctness as _answer_correctness,
        )
    except ImportError:
        raise ImportError(
            "ragas 가 설치되어 있지 않습니다.\n"
            "  pip install ragas"
        )

    if metrics is None:
        metrics = [
            _faithfulness,
            _answer_relevancy,
            _context_precision,
            _context_recall,
            _answer_correctness,
        ]

    # LLM / 임베딩을 각 metric 에 주입
    for m in metrics:
        if llm is not None and hasattr(m, "llm"):
            m.llm = llm
        if embeddings is not None and hasattr(m, "embeddings"):
            m.embeddings = embeddings

    # EvaluationDataset 구성
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=ref,
        )
        for q, a, ctx, ref in zip(questions, answers, contexts, references)
    ]
    dataset = EvaluationDataset(samples=samples)

    # 평가 실행
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    scores = result.to_pandas()
    metric_names = [
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "answer_correctness",
    ]

    output = {}
    for name in metric_names:
        if name in scores.columns:
            output[name] = float(scores[name].mean())
        else:
            output[name] = None

    # RAGAS Score: None 값을 제외한 지표들의 평균
    valid = [v for v in output.values() if v is not None]
    output["ragas_score"] = sum(valid) / len(valid) if valid else None

    return output
