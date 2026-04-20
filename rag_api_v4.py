# rag_api_v3.py에 비용 절감, 속도 향상, 토큰 리밋 방어 리팩토링 버전
# 스레드 병렬 처리 도입
"""
검색 후보 수집
→ 중복 제거
→ _pre_filter 적용
→ 필터 결과 있으면 필터된 후보만 rerank
→ 필터 결과 없으면 원본 후보로 fallback
→ top_n=5 rerank
→ contexts 포장
"""

import math
import uuid
import os
import concurrent.futures
from typing import List, Dict, Any, Optional

# 내부 모듈 임포트
from src.modules.router import route_query
from src.modules.reformulator import reformulate_query
from src.modules.hyde import generate_hyde_document
from src.modules.compressor import compress_document
from src.modules.evaluator import evaluate_contexts
from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker_debug import rerank_and_score

SCORE_THRESHOLD = 0.25  # 랭킹 점수 임계값
PASS_THROUGH_THRESHOLD = 0.85 # 프리패스 기준 점수


# 사전 필터 함수 (rerank 전에 조건 만족 문서만 선별)
# Document 객체의 metadata를 직접 참조
def _pre_filter(candidates: list, filters: dict) -> list:
    if not filters:
        return candidates
 
    result = candidates
 
    # 최소 사업금액 필터
    if "min_budget" in filters:
        result = [doc for doc in result
                  if doc.metadata.get("사업 금액") and
                  doc.metadata.get("사업 금액") >= filters["min_budget"]]
 
    # 최대 사업금액 필터
    if "max_budget" in filters:
        result = [doc for doc in result
                  if doc.metadata.get("사업 금액") and
                  doc.metadata.get("사업 금액") <= filters["max_budget"]]
 
    # 마감일 이전 필터
    if "deadline_before" in filters:
        result = [doc for doc in result
                  if doc.metadata.get("입찰 참여 마감일") and
                  str(doc.metadata.get("입찰 참여 마감일")) <= str(filters["deadline_before"])]
 
    # 마감일 이후 필터
    if "deadline_after" in filters:
        result = [doc for doc in result
                  if doc.metadata.get("입찰 참여 마감일") and
                  str(doc.metadata.get("입찰 참여 마감일")) >= str(filters["deadline_after"])]
 
    return result


def get_rag_context(query: str, chat_history: Optional[List[Dict[str, str]]] = None, alpha: float = 0.4) -> Dict[str, Any]:
    """
    [기능] Bid Mind Advanced RAG 메인 파이프라인 (v3: Multi-Query + Hybrid Search)
    """
    # 파이썬 Mutable Default Argument 버그 방지
    if chat_history is None:
        chat_history = []
        
    print(f"\n" + "="*60)
    print(f"[RAG API 가동] 사용자 질의 접수: '{query}'")
    print("="*60)

    # ==================================================
    # Step 1: 라우터 (Semantic Routing)
    # ==================================================
    route_decision = route_query(query)
    if route_decision == "CHAT":
        print("[Step 1: 라우터] 일상 대화로 감지 -> DB 검색 프로세스 생략")
        return {
            "question": query, 
            "contexts": [], 
            "status": "CHITCHAT", 
            "chat_history": chat_history
        }

    # ==================================================
    # Step 2: Step 2: 질문 분석 (Multi-Query + Metadata Filters)
    # ==================================================
    # 반환형태가 dict로 바뀌었습니다.
    analysis_result = reformulate_query(query, chat_history)
    search_queries = analysis_result.get("queries", [query])
    extracted_filters = analysis_result.get("filters", {})

    has_metadata_filter = bool(
        extracted_filters.get("min_budget")
        or extracted_filters.get("max_budget")
        or extracted_filters.get("deadline_before")
        or extracted_filters.get("deadline_after")
    )

    search_k = 8 if has_metadata_filter else 6
    top_n = 5


    # ==================================================
    # Step 3 & 4: 각 쿼리별 하이브리드 필터 검색 (HyDE 제거 버전)
    # ==================================================
    all_candidates = [] # 모든 쿼리에서 찾아온 문서들을 모아둘 임시 바구니
    
    # 쪼개진 쿼리들을 하나씩 순회하며 각각 검색을 돌립니다. (for loop)
    for sq in search_queries:

        # ❌ [기존 코드 삭제] 가상 문서 생성 생략 (API 호출 비용 및 1~2초 시간 절약)
        # hyde_doc = generate_hyde_document(sq)
        
        # 2. [하이브리드 검색] FAISS와 BM25를 동시 가동합니다.
        # - semantic_query: 벡터 검색(FAISS)용. 긴 문맥(sq)을 넣어 의미를 찾습니다.
        # - keyword_query: 키워드 검색(BM25)용. 짧고 명확한 단어(sq)를 넣어 고유명사를 찾습니다.
        candidates = retrieve_candidates(
            semantic_query=sq, 
            keyword_query=sq,
            filters=extracted_filters,
            k=search_k,
            alpha=alpha
        )
        all_candidates.extend(candidates) # 찾은 문서들을 바구니에 와르르 쏟아붓습니다.
        print(f"   [검색 완료] 쿼리 '{sq}' ➡️ 필터링된 후보 문서 {len(candidates)}개 확보")

    # [중복 제거 로직]
    # "A 정보" 검색과 "B 정보" 검색이 우연히 같은 PDF 파일을 가져올 수 있습니다.
    # 똑같은 문서를 LLM이 두 번 읽지 않도록 중복을 제거(Deduplication)합니다.
    unique_candidates = []
    seen_contents = set()
    for doc in all_candidates:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_candidates.append(doc)
            
    print(f"[Step 4: 다중 검색 병합 완료] 총 {len(unique_candidates)}개의 고유 문서 추출")

    # ==================================================
    # Step 4-1: Pre-filter + Fallback
    # 필터 조건이 있으면 조건 만족 문서만 rerank 대상으로 사용
    # 조건 만족 문서가 없으면 원래 후보로 fallback
    # ==================================================
    used_pre_filter = False  # fallback된 원본후보까지 threshold를 면제하지 않도록, 실제 _pre_filter를 통과한 경우에만 threshold 예외를 줌.
    # 예외주는 이유: _pre_filter를 통과한 문서는 조건이 이미 메타데이터로 검증됐기 때문에 reranker 점수가 낮아도 신뢰할 수 있지만, fallback 문서는 조건 검증 없이 들어온 문서라 threshold로 최소 품질을 보장해야하기 때문.

    if has_metadata_filter:
        filtered_candidates = _pre_filter(unique_candidates, extracted_filters)

        if filtered_candidates:
            print(f"[Pre-filter 완료] 조건 만족 문서 {len(filtered_candidates)}개 → rerank 진행")
            rerank_input = filtered_candidates
            used_pre_filter = True
        else:
            print(f"[Pre-filter 결과 없음] 원래 후보 {len(unique_candidates)}개로 fallback")
            rerank_input = unique_candidates
    else:
        rerank_input = unique_candidates


    # ==================================================
    # Step 4-2: 통합 2차 재정렬 (Reranking)
    # ==================================================
    # 쪼개졌던 쿼리들을 다시 하나의 문자열로 이어 붙입니다. ("A 정보, B 정보")
    # 재정렬 모듈과 압축 모듈은 질문 전체의 맥락을 알아야 하기 때문입니다.
    combined_search_query = ", ".join(search_queries)

    if used_pre_filter:  # True인 경우에만 combined_search_query에 금액/날짜 조건 표현을 추가하는 것을 적용
                         # reranker에 넘어가는 쿼리에 금액/날짜 표현을 앞에 붙여서 reranker가 조건을 인식하고 더 높은 점수를 주도록 유도
        filter_expressions = []
        
        if extracted_filters.get("min_budget"):
            budget = extracted_filters["min_budget"] // 100000000
            filter_expressions.append(f"{budget}억 이상")
        
        if extracted_filters.get("max_budget"):
            budget = extracted_filters["max_budget"] // 100000000
            filter_expressions.append(f"{budget}억 이하")
        
        if extracted_filters.get("deadline_before"):
            filter_expressions.append(f"{extracted_filters['deadline_before'][:10]} 이전 마감")
        
        if extracted_filters.get("deadline_after"):
            filter_expressions.append(f"{extracted_filters['deadline_after'][:10]} 이후 마감")
        
        if filter_expressions:
            combined_search_query = ", ".join(filter_expressions) + ", " + combined_search_query
    
    # 중복이 제거된 문서들을 모아놓고, LLM(또는 Cross-Encoder)을 통해 최종 등수를 매깁니다.
    # 비교 질문일 경우 A, B 양쪽 문서가 모두 포함되어야 하므로 top_n을 5로 넉넉히 잡습니다.
    top_docs = rerank_and_score(combined_search_query, rerank_input, top_n=min(top_n, len(rerank_input))) # rerank_input 문서 수가 5개보다 적을 때도 안전
    print(f"[Step 4-2: 랭킹 완료] 최상위 {len(top_docs)}개 문서 추출 및 유사도 계산")

    # ==================================================
    # Step 5: 문맥 압축 및 필터링 (Context Compression)
    # ==================================================
    contexts = []
    print(f"[Step 5: 문맥 포장] LLM 압축 생략. 원본 텍스트 직접 전달...")

    for score, doc in top_docs:
        if score < SCORE_THRESHOLD and not used_pre_filter:
            continue
            
        original_text = doc.page_content  
        
        contexts.append({
            "chunk_id": f"bid_{uuid.uuid4().hex[:8]}",
            "text": original_text, # 압축하지 않고 원본 그대로 통과!
            "source_file": doc.metadata.get("source", "unknown"),
            "organization": doc.metadata.get("발주 기관", "알 수 없음"),
            "project_name": doc.metadata.get("사업명", "알 수 없음"),
            "budget": doc.metadata.get("사업 금액", None),        # ← 추가
            "deadline": doc.metadata.get("입찰 참여 마감일", None), # ← 추가
            "summary": original_text[:100].replace("\n", " ") + "...",
            "score": math.trunc(score * 100) / 100,  # 기존 표시용 점수
            "raw_score": float(score),               # 분석용 원점수
        })
    
    print(f"[포장 완료] 최종 유효 문서 {len(contexts)}개")   

    # ==================================================
    # Step 6: 자가 반성 (Self-RAG Evaluator) 및 가드레일 프리패스
    # ==================================================
    if contexts:
        # 압축을 통과한 문서들 중 가장 높은 신뢰도 점수를 찾습니다.
        max_score = max(c["score"] for c in contexts)
        
        # [프리패스 발동 조건] 최고 점수가 기준치를 넘었는가?
        if max_score >= PASS_THROUGH_THRESHOLD:
            print(f"🚀 [Step 6: 프리패스 발동] 최고 신뢰도 달성 ({max_score} >= {PASS_THROUGH_THRESHOLD})")
            print("   ✅ 확정적 정답으로 간주하여 LLM 팩트 검증을 생략하고 속도를 극대화합니다.")
        else:
            # 기준치 미달 시 평소대로 검열관을 호출합니다.
            print(f"[Step 6: 자가 반성] 최고 신뢰도({max_score}) 안정권 미달. 팩트 체크를 진행합니다...")
            if not evaluate_contexts(combined_search_query, contexts):
                print("    [Self-RAG 경고] 문서 보강 필요. 환각 방지를 위해 결과 초기화.")
                contexts = []
            else:
                print("   ✅ [Self-RAG 통과] 팩트 검증 완료.")
    else:
        print(f" [Step 6: 자가 반성] 전달할 문서가 없어 검열 생략.")

    print("="*60)
    print(f" [RAG API 완료] 최종 유효 문서 {len(contexts)}개 반환")
    
    # --------------------------------------------------
    # 최종 결과 반환 (LLM 파트 전달용)
    # --------------------------------------------------
    return {
        "original_query": query,  # UI 화면 표시용 (사용자가 친 그대로)
        "question": combined_search_query, # LLM 팀원에게는 완벽해진 질문을 넘김
        "contexts": contexts,
        "status": "SEARCH_SUCCESS" if contexts else "NO_INFO",
        "chat_history": chat_history
    }