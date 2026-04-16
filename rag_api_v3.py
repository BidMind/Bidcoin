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
    # Step 3 & 4: 각 쿼리별 가상 문서 생성 및 하이브리드 필터 검색
    # ==================================================
    all_candidates = [] # 모든 쿼리에서 찾아온 문서들을 모아둘 임시 바구니
    
    # 쪼개진 쿼리들을 하나씩 순회하며 각각 검색을 돌립니다. (for loop)
    for sq in search_queries:
        # 1. [HyDE 적용] 각 쿼리에 대해 가상의 완벽한 정답 문서(hyde_doc)를 만들어냅니다.
        hyde_doc = generate_hyde_document(sq)
        
        # 2. [하이브리드 검색] FAISS와 BM25를 동시 가동합니다.
        # - semantic_query: 벡터 검색(FAISS)용. 긴 문맥(hyde_doc)을 넣어 의미를 찾습니다.
        # - keyword_query: 키워드 검색(BM25)용. 짧고 명확한 단어(sq)를 넣어 고유명사를 찾습니다.
        candidates = retrieve_candidates(
            semantic_query=hyde_doc, 
            keyword_query=sq,
            filters=extracted_filters, # 필터 장착!
            k=10
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
    print(f"[Step 5: 문맥 압축 가동]")
    for score, doc in top_docs:
        if score < SCORE_THRESHOLD:
            source_file = doc.metadata.get("source", "unknown")
            print(f"   [필터링] 점수 미달: {source_file} ({score:.2f} < {SCORE_THRESHOLD})")
            continue
            
        original_text = doc.page_content
        source_file = doc.metadata.get("source", "unknown")
        
        compressed_text = compress_document(combined_search_query, original_text)
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

    # ==================================================
    # Step 6: 자가 반성 (Self-RAG Evaluator)
    # ==================================================
    if contexts:
        print(f"[Step 6: 자가 반성] 최종 팩트 체크 진행 중...")
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