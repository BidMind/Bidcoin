# rag_api_v3.py에 비용 절감, 속도 향상, 토큰 리밋 방어 리팩토링 버전
# 스레드 병렬 처리 도입

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

SCORE_THRESHOLD = 0.6  # 랭킹 점수 임계값
PASS_THROUGH_THRESHOLD = 0.85 # 프리패스 기준 점수

def get_rag_context(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
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
            filters=extracted_filters, # 필터 장착!
            k=6
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
    # Step 4-2: 통합 2차 재정렬 (Reranking)
    # ==================================================
    # 쪼개졌던 쿼리들을 다시 하나의 문자열로 이어 붙입니다. ("A 정보, B 정보")
    # 재정렬 모듈과 압축 모듈은 질문 전체의 맥락을 알아야 하기 때문입니다.
    combined_search_query = ", ".join(search_queries)
    
    # 중복이 제거된 문서들을 모아놓고, LLM(또는 Cross-Encoder)을 통해 최종 등수를 매깁니다.
    # 비교 질문일 경우 A, B 양쪽 문서가 모두 포함되어야 하므로 top_n을 5로 넉넉히 잡습니다.
    top_docs = rerank_and_score(combined_search_query, unique_candidates, top_n=5) 
    print(f"[Step 4-2: 랭킹 완료] 최상위 {len(top_docs)}개 문서 추출 및 유사도 계산")

    # ==================================================
    # Step 5: 문맥 압축 및 필터링 (Context Compression)
    # ==================================================
    contexts = []
    print(f"[Step 5: 문맥 압축 가동] 비동기 병렬 처리 시작...")

    # 1. 각각의 스레드(작업자)가 실행할 단일 압축 함수 정의
    def _compress_worker(doc_info):
        score, doc = doc_info
        if score < SCORE_THRESHOLD:
            return None
            
        original_text = doc.page_content
        # LLM API 호출 지점 
        compressed_text = compress_document(combined_search_query, original_text)
        
        if compressed_text == "PASS":
            return None    
        
        return {
            "chunk_id": f"bid_{uuid.uuid4().hex[:8]}",
            "text": compressed_text,
            "source_file": doc.metadata.get("source", "unknown"),
            "organization": doc.metadata.get("발주 기관", "알 수 없음"),
            "project_name": doc.metadata.get("사업명", "알 수 없음"),
            "summary": compressed_text[:100].replace("\n", " ") + "...",
            "score": math.trunc(score * 100) / 100
        }

    # 2. 최대 5명의 작업자(Thread)를 고용하여 동시에 쫙 뿌립니다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # map 함수를 사용해 top_docs의 문서들을 _compress_worker에 병렬로 던집니다.
        results = list(executor.map(_compress_worker, top_docs))

    # 3. None 값(임계값 미달 또는 PASS된 문서)을 깔끔하게 제거합니다.
    contexts = [res for res in results if res is not None]
    
    print(f"    [압축 성공] 병렬 처리 완료. 최종 유효 문서 {len(contexts)}개 추출")        

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