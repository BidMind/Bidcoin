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

def get_rag_context2(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    [기능] Bid Mind Advanced RAG 메인 파이프라인
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
    # Step 4: 1차 검색 및 2차 재정렬 (Retrieval & Reranking)
    # --------------------------------------------------
    # 주의: 1차 검색은 길게 뻗은 '가상 문서'로, 2차 팩트 랭킹은 깔끔한 '재구성된 질문'으로 진행합니다.
    candidates = retrieve_candidates(hyde_doc, original_query=query, k=10)  # 함수인자 수정
    top_docs = rerank_and_score(search_query, candidates, top_n=3)  
    print(f"[Step 4: 검색 & 랭킹] 최상위 문서 추출 및 유사도 계산 완료")

    # --------------------------------------------------
    # Step 5: 문맥 압축 및 필터링 (Context Compression)
    # --------------------------------------------------
    contexts = []
    print(f"[Step 5: 문맥 압축 가동]")
    for score, doc in top_docs:
        # 1차 가드레일 (수학적 점수)
        if score < SCORE_THRESHOLD:
            source_file = doc.metadata.get("source", "unknown")
            text_preview = doc.page_content[:100].replace("\n", " ") + "..."  # 간단한 텍스트 미리보기
            print("\n" + "-"*60)
            print(f" [Debug] 문서 필터링 됨 (커트라인 미달)")
            print(f" 출처: {source_file}")
            print(f" 점수: {score:.2f} (기준: {SCORE_THRESHOLD})")
            print(f" 미리보기: {text_preview}")
            print("\n" + "-"*60)
            continue
            
        original_text = doc.page_content
        source_file = doc.metadata.get("source", "unknown")
        
        # 2차 가드레일 (시맨틱 필터링)
        compressed_text = compress_document(search_query, original_text)
        if compressed_text == "PASS":
            print(f"   [내용 없음 버림] {source_file}")
            continue

        # 파일명 추출
        clean_name = os.path.splitext(source_file)[0]  # 확장자 제거

        # 확장자 제거, (_) 기준으로 기관/사업명 유추
        parts = clean_name.split("_")
        
        contexts.append({
            "chunk_id": f"bid_{uuid.uuid4().hex[:8]}", # 고유한 chunk ID 생성
            "text": compressed_text, # 문서 내용
            "source_file": source_file, # 원본 파일명
            "organization": parts[0] if len(parts) > 0 else "unknown", # 기관명 유추
            "project_name": parts[1] if len(parts) > 1 else clean_name, # 사업명 유추
            "summary": compressed_text[:100].replace("\n", " ") + "...", # 간단한 요약 (앞 100자)
            "score": math.trunc(score * 100) / 100 # 점수는 소수점 둘째 자리까지 표현
        })
        print(f"    [압축 성공] {source_file} ({len(original_text)}자 ➡️ {len(compressed_text)}자)")

    # --------------------------------------------------
    # Step 6: 자가 반성 (Self-RAG Evaluator)
    # --------------------------------------------------
    if contexts:
        print(f"[Step 6: 자가 반성] 최종 팩트 체크 진행 중...")
        if not evaluate_contexts(search_query, contexts):
            print("    [Self-RAG 경고] 문서 내용은 있으나 질문에 완벽히 답하기 부족함. 환각 방지를 위해 결과 초기화.")
            contexts = []
        else:
            print("   ✅ [Self-RAG 통과] 완벽한 답변을 위한 팩트 검증 완료.")
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