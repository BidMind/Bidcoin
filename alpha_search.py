# qeustion_sets2.py 이용
# alpha_search.py — α 가중치 튜닝 실험 스크립트
#
# 사용법:
#   python alpha_search.py                        → 전체 질문으로 α 실험 실행
#   python alpha_search.py --category 의미기반    → 특정 카테고리만 실험
#   python alpha_search.py --category 키워드기반  → 특정 카테고리만 실험
#   python alpha_search.py --category 복합모호    → 특정 카테고리만 실험

from rag_api_v4 import get_rag_context
from src.evaluation.evaluator import Evaluator
from src.evaluation import metrics
from src.evaluation.question_sets2 import get_by_category, get_all  # ← 변경


def make_retriever(alpha: float):
    def _retriever(query: str) -> list[dict]:
        """
        seen에 이미 본 source_file을 기록해두고, 같은 파일이 또 나오면 건너뜀
        같은 파일에서 청크가 여러개 나오며 평가지표의 분모가 잘못 계산되는 문제 방지
        """
        raw = get_rag_context(query, [], alpha=alpha)
        seen = set()
        result = []
        for ctx in raw.get("contexts", []):
            source = ctx.get("source_file", "unknown")
            if source not in seen:
                seen.add(source)
                result.append({"id": source, "text": ctx.get("text", "")})
        return result
    return _retriever


def run_experiment(category: str | None = None, alphas: list[float] | None = None):
    if alphas is None:
        alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    question_items = get_by_category(category) if category else get_all()
    if not question_items:
        print(f"\n실험 대상 질문이 없습니다. category={category}")
        print("사용 가능한 카테고리: 의미기반 / 키워드기반 / 복합모호")  # ← 변경
        return None

    print(f"\n실험 대상: {len(question_items)}개 질문"
          + (f" (카테고리: {category})" if category else " (전체)"))

    rows = []
    for alpha in alphas:
        evaluator = Evaluator(retriever=make_retriever(alpha), top_k=5)
        results = evaluator.run(question_items)
        agg = metrics.aggregate_metrics(results)
        rows.append((alpha, agg))

    best = {"alpha": None, "mrr": -1}
    print(f"\n{'='*60}")
    print(f"  α 실험 결과 (카테고리: {category or '전체'})")
    print(f"{'='*60}")
    print(f"{'α':>5} | {'hit_rate':>8} | {'recall':>7} | {'mrr':>6} | {'ndcg':>6} | {'precision':>9}")
    print("-" * 55)

    for alpha, agg in rows:
        print(
            f"{alpha:>5} | "
            f"{agg['hit_rate']:>8.3f} | "  # 정답 문서를 놓치지 않는지
            f"{agg['recall']:>7.3f} | "    # 정답 문서가 여러개일 때 몇개 건지는지
            f"{agg['mrr']:>6.3f} | "       # 정답 문서가 상위에 빨리 뜨는지
            f"{agg['ndcg']:>6.3f} | "      # 상위권 전체 품질이 좋은지
            f"{agg['precision']:>9.3f}"    # 가져온 것 중 정답 비율
        )
        if agg["mrr"] > best["mrr"]:
            best = {"alpha": alpha, **agg}

    print("-" * 55)
    print(f"\n최적 α: {best['alpha']}  (MRR 기준)")
    print(f"  hit_rate={best['hit_rate']:.3f}  recall={best['recall']:.3f}"
          f"  mrr={best['mrr']:.3f}  ndcg={best['ndcg']:.3f}  precision={best['precision']:.3f}")
    print(f"{'='*60}\n")

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None,
                        help="특정 카테고리만 평가 (의미기반 / 키워드기반 / 복합모호)")  # ← 변경
    args = parser.parse_args()

    run_experiment(category=args.category)

    # python alpha_search.py --category 의미기반  (빠르게 확인)
    # python alpha_search.py                      (전체 20개로 검증)