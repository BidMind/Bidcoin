# # alpha_search.py — α 가중치 튜닝 실험 스크립트
# #
# # 사용법:
# #   python alpha_search.py                      → 전체 질문으로 α 실험 실행
# #   python alpha_search.py --category 금액      → 특정 카테고리만 실험
# #   python alpha_search.py --category 사업내용  → 특정 카테고리만 실험 


# from rag_api_v4 import get_rag_context
# from src.evaluation.evaluator import Evaluator
# from src.evaluation import metrics
# from src.evaluation.question_sets import get_by_category, get_all
# from src.evaluation.question_sets2 import get_by_category, get_all



# # relevant_ids(공고번호) → source_file 매핑 테이블
# ID_TO_SOURCE = {
#     "고려대학교_차세대 포털·학사 정보시스템 구축사업_고려대학교": "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
#     "20240430896":  "경상북도 봉화군_봉화군 재난통합관리시스템 고도화 사업(협상)(긴급).hwp",
#     "20241001798":  "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.hwp",
#     "R25BK00559883": "한국전기안전공사_전기안전 관제시스템 보안 모듈 개발 용역.hwp",
#     "20240430918":  "인천광역시_도시계획위원회 통합관리시스템 구축용역.hwp",
#     "20241002912":  "한국연구재단_2024년 대학산학협력활동 실태조사 시스템(UICC) 기능개선.hwp",
#     "R25BK00564730": "재단법인충북연구원_GIS통계 기반 재난안전데이터 분석ㆍ관리 시스템 구.hwp",
#     "20240821893":  "국방과학연구소_대용량 자료전송시스템 고도화.hwp",
#     "20240812818":  "(사）한국대학스포츠협의회_KUSF 체육특기자 경기기록 관리시스템 개발.hwp",
#     "20240827859":  "한국생산기술연구원_EIP3.0 고압가스 안전관리 시스템 구축 용역.hwp",
#     "20240821865":  "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
#     "20240815487":  "한국사학진흥재단_대학재정정보시스템(기본재산 및 기채 사후관리) 고.hwp",
#     "한국수자원공사_건설통합시스템(CMS) 고도화_한국수자원공사": "한국수자원공사_건설통합시스템(CMS) 고도화.hwp",
# }

# def convert_relevant_ids(question_items: list[dict]) -> list[dict]:
#     """relevant_ids를 source_file 형식으로 변환한다."""
#     converted = []
#     for item in question_items:
#         new_item = dict(item)
#         new_item["relevant_ids"] = [
#             ID_TO_SOURCE.get(rid, rid)
#             for rid in item["relevant_ids"]
#         ]
#         converted.append(new_item)
#     return converted


# def make_retriever(alpha: float):
#     def _retriever(query: str) -> list[dict]:
#         """
#         seen에 이미 본 source_file을 기록해두고, 같은 파일이 또 나오면 건너뜀
#         같은 파일에서 청크가 여러개 나오며 평가지표의 분모가 잘못 계산되는 문제 방지
#         """
#         raw = get_rag_context(query, [], alpha=alpha)
#         seen = set()  
#         result = []
#         for ctx in raw.get("contexts", []):
#             source = ctx.get("source_file", "unknown")
#             if source not in seen:
#                 seen.add(source)
#                 result.append({"id": source, "text": ctx.get("text", "")})
#         return result
#     return _retriever


# def run_experiment(category: str | None = None, alphas: list[float] | None = None):
#     if alphas is None:
#         alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

#     question_items = get_by_category(category) if category else get_all()
#     question_items = convert_relevant_ids(question_items)  # 매핑변환 적용
#     if not question_items:
#         print(f"\n실험 대상 질문이 없습니다. category={category}")
#         print("사용 가능한 카테고리: 금액 / 기간 / 기관 / 복합 / 사업내용")
#         return None
    
#     print(f"\n실험 대상: {len(question_items)}개 질문"
#           + (f" (카테고리: {category})" if category else " (전체)"))

#     rows = []
#     for alpha in alphas:
#         evaluator = Evaluator(retriever=make_retriever(alpha), top_k=5)
#         results = evaluator.run(question_items)
#         agg = metrics.aggregate_metrics(results)
#         rows.append((alpha, agg))
    
#     # 모두 끝난 후 표 한번에 출력
#     best = {"alpha": None, "mrr": -1}
#     print(f"\n{'='*60}")
#     print(f"  α 실험 결과 (카테고리: {category or '전체'})")
#     print(f"{'='*60}")
#     print(f"{'α':>5} | {'hit_rate':>8} | {'recall':>7} | {'mrr':>6} | {'ndcg':>6} | {'precision':>9}")
#     print("-" * 55)

#     for alpha, agg in rows:
#         print(
#             f"{alpha:>5} | "
#             f"{agg['hit_rate']:>8.3f} | " # 정답 문서를 놓치지 않는지
#             f"{agg['recall']:>7.3f} | "   # 정답 문서가 여러개일 때 몇개 건지는지
#             f"{agg['mrr']:>6.3f} | "      # 정답 문서가 상위에 빨리 뜨는지
#             f"{agg['ndcg']:>6.3f} | "     # 상위권 전체 품질이 좋은지
#             f"{agg['precision']:>9.3f}"
#         )
#         if agg["mrr"] > best["mrr"]:
#             best = {"alpha": alpha, **agg}

#     print("-" * 55)
#     print(f"\n최적 α: {best['alpha']}  (MRR 기준)")
#     print(f"  hit_rate={best['hit_rate']:.3f}  recall={best['recall']:.3f}"
#           f"  mrr={best['mrr']:.3f}  ndcg={best['ndcg']:.3f}  precision={best['precision']:.3f}")
#     print(f"{'='*60}\n")

#     return best


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--category", type=str, default=None,
#                         help="특정 카테고리만 평가 (금액 / 기간 / 기관 / 복합 / 사업내용)")
#     args = parser.parse_args()

#     # 빠른 실험은 특정 카테고리부터, 확인되면 전체로 확장
#     run_experiment(category=args.category)

#     # python alpha_search.py --category 금액 (빠르게 확인)
#     # python alpha_search.py (전체 34개로 검증)


# ======================================================================
# qeustion_sets2.py 이용
# ======================================================================
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
        print("사용 가능한 카테고리: 금액 /기간 /기관 /복합 /사업내용")  # ← 변경
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
                        help="특정 카테고리만 평가 (금액 /기간 /기관 /복합 /사업내용)")  # ← 변경
    args = parser.parse_args()

    run_experiment(category=args.category)

    # python alpha_search.py --category 의미기반  (빠르게 확인)
    # python alpha_search.py                      (전체 20개로 검증)