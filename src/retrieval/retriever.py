from src.embedding.vector_store import load_db

def retrieve_candidates(query: str, k: int = 10):
    """
    [기능] 빠르지만 정확도가 떨어지는 1차 검색을 수행합니다.
    [흐름] 질문(query) -> 벡터 DB -> DB에서 가장 유사한 k개 문서 반환
    """
    vector_store = load_db()
    # similarity_search: 벡터 DB에서 질문과 가장 유사한 k개의 문서를 반환
    candidates = vector_store.similarity_search(query, k=k)
    return candidates