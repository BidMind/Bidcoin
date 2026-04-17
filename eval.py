# eval.py — RAG 파이프라인 평가 실행 스크립트
#
# 사용법:
#   python eval.py                      → question_sets.py 전체 질문으로 2가지 평가 모두 실행 (기본)
#   python eval.py --category fact      → 특정 카테고리만 평가
#   python eval.py --basic              → 기본 평가만 (빠름, LLM 불필요)
#   python eval.py --judge              → LLM-as-Judge 평가만

from __future__ import annotations

import argparse
import json
from datetime import datetime

from rag_api_v4 import get_rag_context
#from rag_api import get_rag_context
from src.generation.generator import BidCoinGenerator
from src.generation.schemas import RetrievedContext, RetrievalResult
from src.evaluation.evaluator import Evaluator
from src.evaluation.question_sets import get_all, get_by_category

# ─── Generator 1회 초기화 (질문마다 재생성하지 않음) ─────────────────────────
_generator_instance: BidCoinGenerator | None = None

def _get_generator() -> BidCoinGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = BidCoinGenerator()
    return _generator_instance


# ─── Evaluator 래퍼 함수 ─────────────────────────────────────────────────────

# retriever 가 가져온 전체 메타데이터를 generator 에서 재사용하기 위한 캐시.
# Evaluator 는 generator 에 text 문자열만 전달하므로 source_file 등 메타데이터가
# 중간에 유실된다. retriever 호출 시 원본 컨텍스트를 여기에 저장해 두고
# generator 에서 꺼내 쓰면 source_file 이 항상 "eval" 로 고정되는 문제가 해결된다.
_retrieval_cache: dict[str, list[dict]] = {}


def _retriever(query: str) -> list[dict]:
    """
    Evaluator가 요구하는 retriever 시그니처 래퍼.
      (query: str) -> List[{"id": str, "text": str}]

    question_sets.py 의 question 을 그대로 RAG 파이프라인에 전달하고,
    반환된 contexts 의 source_file 을 id 로 사용한다.
    전체 메타데이터는 _retrieval_cache 에 보존해 generator 에서 재사용한다.
    """
    raw = get_rag_context(query, [])
    full_contexts = raw.get("contexts", [])
    _retrieval_cache[query] = full_contexts          # 메타데이터 보존
    return [
        {
            "id":   ctx.get("source_file", ctx.get("chunk_id", "unknown")),
            "text": ctx.get("text", ""),
        }
        for ctx in full_contexts
    ]


def _generator(query: str, contexts: list[str]) -> str:
    """
    Evaluator가 요구하는 generator 시그니처 래퍼.
      (query: str, contexts: List[str]) -> str

    _retriever 가 캐시해 둔 원본 메타데이터를 활용해 RetrievedContext 를 복원한다.
    캐시가 없을 경우(안전망)에만 text 만으로 객체를 생성한다.
    """
    full_contexts = _retrieval_cache.get(query, [])
    if full_contexts:
        ctx_objs = [
            RetrievedContext(
                text=ctx.get("text", ""),
                source_file=ctx.get("source_file", "unknown"),
                chunk_id=ctx.get("chunk_id"),
                organization=ctx.get("organization"),
                project_name=ctx.get("project_name"),
                summary=ctx.get("summary"),
                score=ctx.get("score"),
            )
            for ctx in full_contexts
        ]
    else:
        ctx_objs = [RetrievedContext(text=t, source_file="unknown") for t in contexts]
    retrieval_result = RetrievalResult(question=query, contexts=ctx_objs)
    return _get_generator().generate(retrieval_result).answer


# ─── JSON 저장 헬퍼 ──────────────────────────────────────────────────────────

def _write_json(data: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[JSON 저장 완료] {path}")


# ─── 평가 실행 함수 ───────────────────────────────────────────────────────────

def run_basic(evaluator: Evaluator, question_items: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  [1/2] 기본 평가  (Hit Rate / Precision / Recall / MRR / NDCG / F1 / Faithfulness)")
    print("="*60)
    results = evaluator.run(question_items)
    Evaluator.print_report(results)

    return [
        {
            "id":             r.get("id"),
            "category":       r.get("category", ""),
            "query":          r.get("question"),
            "example_answer": r.get("reference_answer"),
            "answer":         r.get("generated_answer"),
            "evaluation": {
                "hit_rate":    r.get("hit_rate"),
                "precision":   r.get("precision"),
                "recall":      r.get("recall"),
                "mrr":         r.get("mrr"),
                "ndcg":        r.get("ndcg"),
                "f1":          r.get("f1"),
                "faithfulness": r.get("faithfulness"),
            },
        }
        for r in results
    ]


def run_judge(evaluator: Evaluator, question_items: list[dict]) -> list[dict]:
    print("\n" + "="*70)
    print("  [2/2] LLM-as-Judge 평가  (Relevance / Faithfulness / Correctness / Completeness / Context Precision·Recall)")
    print("="*70)
    scores = evaluator.run_llm_judge(question_items)
    Evaluator.print_llm_judge_report(scores)

    return [
        {
            "id":             item.get("id"),
            "category":       item.get("category", ""),
            "query":          item["question"],
            "example_answer": item["answer"],
            "answer":         pq.get("answer"),
            "evaluation": {
                "relevance":         pq.get("relevance"),
                "faithfulness":      pq.get("faithfulness"),
                "correctness":       pq.get("correctness"),
                "completeness":      pq.get("completeness"),
                "context_precision": pq.get("context_precision"),
                "context_recall":    pq.get("context_recall"),
                "reason":            pq.get("reason"),
            },
        }
        for item, pq in zip(question_items, scores.get("per_query", []))
    ]


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BidCoin RAG 파이프라인 평가")
    parser.add_argument("--basic",    action="store_true", help="기본 평가만 실행 (빠름, LLM 불필요)")
    parser.add_argument("--judge",    action="store_true", help="LLM-as-Judge 평가만 실행")
    parser.add_argument("--category", type=str, default=None,
                        help="특정 카테고리만 평가 (fact / condition / summary / compare / recommend / follow_up / refusal / evidence / complex)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="JSON 결과 저장 경로 (기본: eval_results_<timestamp>.json)")
    args = parser.parse_args()

    # question_sets.py 에서 질문 로드
    question_items = get_by_category(args.category) if args.category else get_all()
    print(f"\n평가 질문 수: {len(question_items)}개"
          + (f"  (카테고리: {args.category})" if args.category else "  (전체)"))
    for q in question_items:
        print(f"  [{q['category']}] {q['question'][:60]}")

    # Evaluator 초기화 (generator 는 모듈 레벨에서 1회만 생성)
    evaluator = Evaluator(retriever=_retriever, generator=_generator, top_k=5)

    # JSON 출력 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"eval_results_{timestamp}.json"

    # 단일 모드 플래그가 있으면 해당 평가만, 없으면 2가지 모두 실행
    if args.basic:
        records = run_basic(evaluator, question_items)
        _write_json(records, output_path)
    elif args.judge:
        records = run_judge(evaluator, question_items)
        _write_json(records, output_path)
    else:
        # 기본: basic + LLM-as-Judge 실행 후 eval_type별로 묶어 저장
        basic_records = run_basic(evaluator, question_items)
        judge_records = run_judge(evaluator, question_items)
        _write_json(
            {"basic": basic_records, "judge": judge_records},
            output_path,
        )


if __name__ == "__main__":
    main()
