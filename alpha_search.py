# alpha_search.py — α 가중치 튜닝 실험 스크립트
#
# 사용법:
#   python alpha_search.py                           → 전체 질문으로 α 실험 실행
#   python alpha_search.py --category recommend      → 특정 카테고리만 실험
#   python alpha_search.py --category fact           → 특정 카테고리만 실험 

import os
import sys
from rag_api_v4 import get_rag_context
from src.evaluation.evaluator import Evaluator
from src.evaluation import metrics
from src.evaluation.question_sets import get_by_category, get_all


def make_retriever(alpha: float):
    def _retriever(query: str) -> list[dict]:
        """로그없이 표만 깔끔하게 보기 위함"""
        # 표준 출력 억제
        sys.stdout = open(os.devnull, 'w')
        try:
            raw = get_rag_context(query, [], alpha=alpha)
        finally:
            # 반드시 복구 (에러나도 복구되도록 finally)
            sys.stdout = sys.__stdout__
        return [
            {
                "id":   ctx.get("source_file", "unknown"),
                "text": ctx.get("text", ""),
            }
            for ctx in raw.get("contexts", [])
        ]
    return _retriever


def run_experiment(category: str | None = None, alphas: list[float] | None = None):
    if alphas is None:
        alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    question_items = get_by_category(category) if category else get_all()
    print(f"\n실험 대상: {len(question_items)}개 질문"
          + (f" (카테고리: {category})" if category else " (전체)"))

    rows = []
    for alpha in alphas:
        evaluator = Evaluator(retriever=make_retriever(alpha), top_k=5)
        results = evaluator.run(question_items)
        agg = metrics.aggregate_metrics(results)
        rows.append((alpha, agg))
    
    best = {"alpha": None, "mrr": -1}
    print(f"\n{'α':>5} | {'hit_rate':>8} | {'recall':>7} | {'mrr':>6} | {'ndcg':>6} | {'precision':>9}")
    print("-" * 55)

    for alpha, agg in rows:
        print(
            f"{alpha:>5} | "
            f"{agg['hit_rate']:>8.3f} | "
            f"{agg['recall']:>7.3f} | "
            f"{agg['mrr']:>6.3f} | "
            f"{agg['ndcg']:>6.3f} | "
            f"{agg['precision']:>9.3f}"
        )
        if agg["mrr"] > best["mrr"]:
            best = {"alpha": alpha, **agg}

    print("-" * 55)
    print(f"\n최적 α: {best['alpha']}  (MRR 기준)")
    print(f"  hit_rate={best['hit_rate']:.3f}  recall={best['recall']:.3f}"
          f"  mrr={best['mrr']:.3f}  ndcg={best['ndcg']:.3f}")

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None,
                        help="특정 카테고리만 평가 (fact / condition / summary / compare / recommend / follow_up / refusal / evidence / complex)")
    args = parser.parse_args()

    # 빠른 실험은 recommend(3개)부터, 확인되면 전체로 확장
    run_experiment(category=args.category)

    # python alpha_search.py --category recommend (빠르게 방향 확인)
    # python alpha_search.py (방향 확인되면 전체 34개로 검증)