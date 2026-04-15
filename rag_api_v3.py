import math
import uuid
import os
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

def get_rag_context(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    [기능] Bid Mind Advanced RAG 메인 파이프라인 (v3 하이브리드)
    """
    # 파이썬 Mutable Default Argument 버그 방지
    if chat_history is None:
        chat_history = []
        
    print(f"\n" + "="*60)
    print(f"[RAG API 가동] 사용자 질의 접수: '{query}'")
    print("="*60)

    # --------------------------------------------------
    # Step 1: 라우터 (Semantic Routing)
    # --------------------------------------------------
    route_decision = route_query(query)
    if route_decision == "CHAT":
        print("[Step 1: 라우터] 일상 대화로 감지 -> DB 검색 프로세스 생략")
        return {
            "question": query, 
            "contexts": [], 
            "status": "CHITCHAT", 
            "chat_history": chat_history
        }

    # --------------------------------------------------
    # Step 2: 질문 재구성 (Query Reformulation)
    # --------------------------------------------------
    search_query = reformulate_query(query, chat_history)
    print(f"[Step 2: 질문 재구성] {search_query}")

    # --------------------------------------------------
    # Step 3: HyDE (가상 문서 생성)
    # --------------------------------------------------
    hyde_doc = generate_hyde_document(search_query)
    print(f"[Step 3: HyDE 가상 문서 생성] 검색 명중률 증폭 완료")

    # --------------------------------------------------
    # Step 4: 1차 하이브리드 검색 및 2차 재정렬 (Retrieval & Reranking)
    # --------------------------------------------------
    # - semantic_query: 벡터 검색(FAISS)용. 풍부한 맥락의 hyde_doc 사용.
    # - keyword_query: 키워드 검색(BM25)용. 노이즈가 적은 search_query 사용.
    candidates = retrieve_candidates(
        semantic_query=hyde_doc,
        keyword_query=search_query,
        k=10
    ) 

    top_docs = rerank_and_score(search_query, candidates, top_n=3) 
    print(f"[Step 4: 하이브리드 검색 & 랭킹] 최상위 문서 추출 및 유사도 계산 완료")

    # --------------------------------------------------
    # Step 5: 문맥 압축 및 필터링 (Context Compression)
    # --------------------------------------------------
    contexts = []
    print(f"[Step 5: 문맥 압축 가동]")
    for score, doc in top_docs:
        if score < SCORE_THRESHOLD:
            source_file = doc.metadata.get("source", "unknown")
            print(f"   [필터링] 점수 미달: {source_file} ({score:.2f} < {SCORE_THRESHOLD})")
            continue
            
        original_text = doc.page_content
        source_file = doc.metadata.get("source", "unknown")
        
        compressed_text = compress_document(search_query, original_text)
        if compressed_text == "PASS":
            continue

        clean_name = os.path.splitext(source_file)[0]
        parts = clean_name.split("_")
        
        contexts.append({
            "chunk_id": f"bid_{uuid.uuid4().hex[:8]}",
            "text": compressed_text,
            "source_file": source_file,
            "organization": parts[0] if len(parts) > 0 else "unknown",
            "project_name": parts[1] if len(parts) > 1 else clean_name,
            "summary": compressed_text[:100].replace("\n", " ") + "...",
            "score": math.trunc(score * 100) / 100
        })
        print(f"    [압축 성공] {source_file} ({len(original_text)}자 ➡️ {len(compressed_text)}자)")

    # --------------------------------------------------
    # Step 6: 자가 반성 (Self-RAG Evaluator)
    # --------------------------------------------------
    if contexts:
        print(f"[Step 6: 자가 반성] 최종 팩트 체크 진행 중...")
        if not evaluate_contexts(search_query, contexts):
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
        "question": search_query, # LLM 팀원에게는 완벽해진 질문을 넘김
        "contexts": contexts,
        "status": "SEARCH_SUCCESS" if contexts else "NO_INFO",
        "chat_history": chat_history
    }