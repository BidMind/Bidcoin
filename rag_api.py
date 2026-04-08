import config
import json
import uuid
from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker import rerank_and_score

def get_rag_context(query: str, chat_history: list = []):
    """
    [기능] RAG 시스템의 핵심 함수로, 질문과 대화 이력을 바탕으로 최종적으로 랭킹된 문서 리스트를 반환합니다.
    [흐름] 질문(query) -> 1차 검색 -> 2차 랭킹 -> 최종 문서 리스트 반환
    """
    # 1차 검색: FAISS에서 10개 문서 1차 추출
    candidates = retrieve_candidates(query, k=10)

    # 2차 랭킹: 상위 3개 추출
    top_docs = rerank_and_score(query, candidates, top_n=3)

    # context 리스트 생성: 최종적으로 랭킹된 문서 리스트 반환
    contexts = []
    for score, doc in top_docs:
        # 파일명 추출
        source_file = doc.metadata.get("source", "unknown")

        # 확장자 제거, (_) 기준으로 기관/사업명 유추
        parts = source_file.replace(".pdf", "").replace(".hwp", "").replace(".csv", "").split("_")
        
        contexts.append({
            "chunk_id": f"korea_u_{uuid.uuid4().hex[:2]}", # 고유한 chunk ID 생성
            "text": doc.page_content, # 문서 내용
            "source_file": source_file, # 원본 파일명
            "organization": parts[0] if len(parts) > 0 else "unknown", # 기관명 유추
            "project_name": parts[1] if len(parts) > 1 else "source_file", # 사업명 유추
            "summary": doc.page_content[:100].replace("\n", " ") + "...", # 간단한 요약 (앞 100자)
            "score": round(score, 4) # Sigmoid 점수 (0~1 사이, 소수점 4자리로 반올림)
        })
    # 최종 반환 JSON 구조
    return {
        "question": query,
        "contexts": contexts,
        "chat_history": chat_history
    }