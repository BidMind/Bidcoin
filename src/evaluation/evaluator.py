"""
evaluator.py — RAG 파이프라인 평가 실행기

사용법:
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.question_sets import get_all

    evaluator = Evaluator(retriever=my_retriever, generator=my_generator, top_k=5)

    # 기본 평가 (LLM 불필요)
    results = evaluator.run(get_all())
    evaluator.print_report(results)

    # RAGAS 평가 (LLM 필요)
    ragas_scores = evaluator.run_ragas(get_all())
    print(ragas_scores)

    # LLM-as-Judge 평가 (LLM 필요)
    judge_scores = evaluator.run_llm_judge(get_all())
    evaluator.print_llm_judge_report(judge_scores)

retriever: (query: str) -> List[dict]
    각 dict 는 반드시 'id' 와 'text' 키를 포함해야 한다.
    예) {"id": "20241001798", "text": "[공고번호: ...]\n\n..."}

generator: (query: str, contexts: List[str]) -> str
    검색된 청크들을 컨텍스트로 받아 답변 문자열을 반환한다.
    generator=None 이면 생성 단계를 건너뛰고 검색 지표만 계산한다.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from src.evaluation import metrics as M
from src.evaluation.question_sets import get_all, get_by_category


class Evaluator:
    def __init__(
        self,
        retriever: Callable[[str], List[dict]],
        generator: Optional[Callable[[str, List[str]], str]] = None,
        top_k: int = 5,
    ):
        """
        Args:
            retriever : 쿼리를 받아 {'id', 'text', ...} dict 리스트를 반환하는 함수.
            generator : 쿼리와 컨텍스트 리스트를 받아 답변 문자열을 반환하는 함수.
                        None 이면 검색 지표만 계산한다.
            top_k     : 검색 결과 상위 K개를 평가 대상으로 삼는다.
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    # ------------------------------------------------------------------
    # 단일 쿼리 평가
    # ------------------------------------------------------------------

    def _evaluate_one(self, question_item: dict) -> dict:
        query = question_item["question"]
        relevant_ids = question_item["relevant_ids"]

        # 1. 검색
        t0 = time.time()
        retrieved = self.retriever(query)[: self.top_k]
        retrieval_time = time.time() - t0

        retrieved_ids = [r["id"] for r in retrieved]
        context_chunks = [r["text"] for r in retrieved]

        # 2. 검색 지표 계산
        result = {
            "id": question_item["id"],
            "question": query,
            "category": question_item.get("category", ""),
            "reference_answer": question_item["answer"],
            "retrieved_ids": retrieved_ids,
            "retrieval_time_sec": round(retrieval_time, 3),
            "hit_rate": M.hit_rate(retrieved_ids, relevant_ids),
            "precision": M.precision_at_k(retrieved_ids, relevant_ids),
            "recall": M.recall_at_k(retrieved_ids, relevant_ids),
            "mrr": M.mrr(retrieved_ids, relevant_ids),
            "ndcg": M.ndcg_at_k(retrieved_ids, relevant_ids),
        }

        # 3. 생성 지표 계산 (generator 가 있을 때만)
        if self.generator is not None:
            t1 = time.time()
            generated_answer = self.generator(query, context_chunks)
            generation_time = time.time() - t1

            result["generated_answer"] = generated_answer
            result["generation_time_sec"] = round(generation_time, 3)
            result["f1"] = M.token_overlap_f1(generated_answer, question_item["answer"])
            result["faithfulness"] = M.answer_in_context(question_item["answer"], context_chunks)
        else:
            result["generated_answer"] = None
            result["f1"] = 0.0
            result["faithfulness"] = M.answer_in_context(question_item["answer"], context_chunks)

        return result

    # ------------------------------------------------------------------
    # 전체 평가 실행
    # ------------------------------------------------------------------

    def run(
        self,
        question_items: Optional[List[dict]] = None,
        category: Optional[str] = None,
    ) -> List[dict]:
        """
        Args:
            question_items : 평가할 질문 리스트. None 이면 전체 질문 셋 사용.
            category       : 특정 카테고리만 평가하고 싶을 때 지정.

        Returns:
            per-query 결과 dict 리스트.
        """
        if question_items is None:
            question_items = get_by_category(category) if category else get_all()

        results = []
        for item in question_items:
            try:
                result = self._evaluate_one(item)
            except Exception as e:
                result = {
                    "id": item.get("id", "?"),
                    "question": item.get("question", ""),
                    "error": str(e),
                    "hit_rate": 0.0, "precision": 0.0, "recall": 0.0,
                    "mrr": 0.0, "ndcg": 0.0, "f1": 0.0, "faithfulness": 0.0,
                }
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # 리포트 출력
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(results: List[dict]) -> None:
        """평가 결과를 콘솔에 출력한다."""
        agg = M.aggregate_metrics(results)

        print("=" * 60)
        print(f"  RAG 평가 결과 (총 {agg.get('n_queries', 0)}개 질문)")
        print("=" * 60)

        # 지표별 집계
        print(f"  Hit Rate    : {agg.get('hit_rate', 0):.4f}")
        print(f"  Precision@K : {agg.get('precision', 0):.4f}")
        print(f"  Recall@K    : {agg.get('recall', 0):.4f}")
        print(f"  MRR         : {agg.get('mrr', 0):.4f}")
        print(f"  NDCG@K      : {agg.get('ndcg', 0):.4f}")
        print(f"  Token F1    : {agg.get('f1', 0):.4f}")
        print(f"  Faithfulness: {agg.get('faithfulness', 0):.4f}")
        print("-" * 60)

        # 카테고리별 Hit Rate
        categories = sorted(set(r.get("category", "") for r in results))
        if categories:
            print("  카테고리별 Hit Rate:")
            for cat in categories:
                cat_results = [r for r in results if r.get("category") == cat]
                cat_hr = sum(r["hit_rate"] for r in cat_results) / len(cat_results)
                print(f"    {cat:<12}: {cat_hr:.4f}  ({len(cat_results)}건)")
        print("-" * 60)

        # 실패 항목
        failed = [r for r in results if r.get("hit_rate", 0) == 0.0]
        if failed:
            print(f"  검색 실패 ({len(failed)}건):")
            for r in failed:
                print(f"    [{r['id']}] {r['question'][:45]}...")
        print("=" * 60)

    @staticmethod
    def to_dataframe(results: List[dict]):
        """결과를 pandas DataFrame 으로 변환한다."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas 가 설치되어 있지 않습니다: pip install pandas")

        rows = []
        for r in results:
            rows.append({
                "id": r.get("id"),
                "category": r.get("category"),
                "question": r.get("question"),
                "reference_answer": r.get("reference_answer"),
                "generated_answer": r.get("generated_answer"),
                "hit_rate": r.get("hit_rate", 0.0),
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "mrr": r.get("mrr", 0.0),
                "ndcg": r.get("ndcg", 0.0),
                "f1": r.get("f1", 0.0),
                "faithfulness": r.get("faithfulness", 0.0),
                "retrieval_time_sec": r.get("retrieval_time_sec"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # RAGAS 평가 (LLM 필요)
    # ------------------------------------------------------------------

    def run_ragas(
        self,
        question_items: Optional[List[dict]] = None,
        category: Optional[str] = None,
        llm=None,
        embeddings=None,
        ragas_metrics: Optional[List] = None,
    ) -> dict:
        """
        RAGAS 5개 지표를 계산한다. generator 가 반드시 설정되어 있어야 한다.

        Args:
            question_items : 평가할 질문 리스트. None 이면 전체 질문 셋 사용.
            category       : 특정 카테고리만 평가하고 싶을 때 지정.
            llm            : RAGAS LLM 객체. None 이면 OPENAI_API_KEY 환경변수로 자동 설정.
                             예) from langchain_openai import ChatOpenAI
                                 llm = ChatOpenAI(model="gpt-4o-mini")
                             예) from langchain_anthropic import ChatAnthropic
                                 llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
            embeddings     : RAGAS 임베딩 객체. None 이면 OpenAI 임베딩 자동 사용.
            ragas_metrics  : 계산할 RAGAS Metric 리스트. None 이면 5개 전체 사용.

        Returns:
            {"faithfulness": float, "answer_relevancy": float,
             "context_precision": float, "context_recall": float,
             "answer_correctness": float, "ragas_score": float}

        사용 예시:
            import os
            os.environ["OPENAI_API_KEY"] = "sk-..."

            evaluator = Evaluator(retriever=my_retriever, generator=my_generator)
            ragas_scores = evaluator.run_ragas()
            print(ragas_scores)
        """
        if self.generator is None:
            raise ValueError(
                "run_ragas() 는 generator 가 필요합니다.\n"
                "Evaluator(retriever=..., generator=my_generator) 로 초기화하세요."
            )

        if question_items is None:
            question_items = get_by_category(category) if category else get_all()

        questions, answers, contexts, references = [], [], [], []

        for item in question_items:
            query = item["question"]
            retrieved = self.retriever(query)[: self.top_k]
            ctx_chunks = [r["text"] for r in retrieved]
            answer = self.generator(query, ctx_chunks)

            questions.append(query)
            answers.append(answer)
            contexts.append(ctx_chunks)
            references.append(item["answer"])

        return M.ragas_evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            references=references,
            llm=llm,
            embeddings=embeddings,
            metrics=ragas_metrics,
        )

    # ------------------------------------------------------------------
    # LLM-as-Judge 평가
    # ------------------------------------------------------------------

    def run_llm_judge(
        self,
        question_items: Optional[List[dict]] = None,
        category: Optional[str] = None,
        llm=None,
        criteria: Optional[List] = None,
    ) -> dict:
        """
        LLM-as-Judge 방식으로 생성 답변 품질을 평가한다.
        generator 가 반드시 설정되어 있어야 한다.

        Args:
            question_items : 평가할 질문 리스트. None 이면 전체 질문 셋 사용.
            category       : 특정 카테고리만 평가하고 싶을 때 지정.
            llm            : LangChain LLM 객체. None 이면 OPENAI_API_KEY 로 gpt-4o-mini 자동 사용.
                             예) from langchain_openai import ChatOpenAI
                                 llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                             예) from langchain_anthropic import ChatAnthropic
                                 llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
            criteria       : 집계할 기준 리스트. None 이면 4개 전체.
                             ["relevance", "faithfulness", "correctness", "completeness"]

        Returns:
            {
                "relevance":    float,      # 0.0 ~ 1.0
                "faithfulness": float,
                "correctness":  float,
                "completeness": float,
                "judge_score":  float,      # 기준 지표 평균
                "per_query":    List[dict]  # 질문별 점수 및 이유
            }

        사용 예시:
            import os
            os.environ["OPENAI_API_KEY"] = "sk-..."

            evaluator = Evaluator(retriever=my_retriever, generator=my_generator)
            judge_scores = evaluator.run_llm_judge()
            Evaluator.print_llm_judge_report(judge_scores)
        """
        if self.generator is None:
            raise ValueError(
                "run_llm_judge() 는 generator 가 필요합니다.\n"
                "Evaluator(retriever=..., generator=my_generator) 로 초기화하세요."
            )

        if question_items is None:
            question_items = get_by_category(category) if category else get_all()

        questions, answers, contexts, references = [], [], [], []

        for item in question_items:
            query = item["question"]
            retrieved = self.retriever(query)[: self.top_k]
            ctx_chunks = [r["text"] for r in retrieved]
            answer = self.generator(query, ctx_chunks)

            questions.append(query)
            answers.append(answer)
            contexts.append(ctx_chunks)
            references.append(item["answer"])

        return M.llm_judge_evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            references=references,
            llm=llm,
            criteria=criteria,
        )

    @staticmethod
    def print_llm_judge_report(judge_scores: dict) -> None:
        """run_llm_judge() 결과를 콘솔에 출력한다."""
        print("=" * 60)
        print("  LLM-as-Judge 평가 결과")
        print("=" * 60)
        labels = {
            "relevance":    "Relevance    (관련성)",
            "faithfulness": "Faithfulness (충실성)",
            "correctness":  "Correctness  (정확성)",
            "completeness": "Completeness (완전성)",
            "judge_score":  "Judge Score  (종합)",
        }
        for key, label in labels.items():
            val = judge_scores.get(key)
            val_str = f"{val:.4f}" if val is not None else "N/A"
            print(f"  {label}: {val_str}")

        per_query = judge_scores.get("per_query", [])
        if per_query:
            print("-" * 60)
            print("  질문별 상세 점수:")
            for i, row in enumerate(per_query, 1):
                q_short = row["question"][:40]
                if row.get("error"):
                    print(f"  {i:>2}. [{q_short}] ERROR: {row['error']}")
                else:
                    r = row.get("relevance")
                    f = row.get("faithfulness")
                    c = row.get("correctness")
                    m = row.get("completeness")
                    fmt = lambda v: f"{v:.2f}" if v is not None else " N/A"
                    print(
                        f"  {i:>2}. Rel={fmt(r)} Fai={fmt(f)} "
                        f"Cor={fmt(c)} Com={fmt(m)}  {q_short}"
                    )
                    if row.get("reason"):
                        print(f"       └ {row['reason']}")
        print("=" * 60)

    @staticmethod
    def print_ragas_report(ragas_scores: dict) -> None:
        """run_ragas() 결과를 콘솔에 출력한다."""
        print("=" * 60)
        print("  RAGAS 평가 결과")
        print("=" * 60)
        labels = {
            "faithfulness":       "Faithfulness      (충실성)",
            "answer_relevancy":   "Answer Relevancy  (답변 관련성)",
            "context_precision":  "Context Precision (컨텍스트 정밀도)",
            "context_recall":     "Context Recall    (컨텍스트 재현율)",
            "answer_correctness": "Answer Correctness(답변 정확도)",
            "ragas_score":        "RAGAS Score       (종합)",
        }
        for key, label in labels.items():
            val = ragas_scores.get(key)
            val_str = f"{val:.4f}" if val is not None else "N/A"
            print(f"  {label}: {val_str}")
        print("=" * 60)
