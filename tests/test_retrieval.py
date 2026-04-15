import os
from src.embedding.keyword_store import load_db, load_bm25

def verify_indices(query: str):
    print(f"\n🔍 테스트 질문: '{query}'")
    print("=" * 60)

    # 1. FAISS (Vector) 테스트
    try:
        print("\n🤖 [1/2] FAISS 벡터 검색 중...")
        vector_db = load_db()
        # k=3 정도로 상위 3개만 확인
        v_results = vector_db.similarity_search(query, k=3)
        for i, doc in enumerate(v_results):
            print(f"   [{i+1}] {doc.metadata.get('source')} | 내용: {doc.page_content[:50]}...")
    except Exception as e:
        print(f"   ❌ FAISS 로드 실패: {e}")

    # 2. BM25 (Keyword) 테스트
    try:
        print("\n⌨️ [2/2] BM25 키워드 검색 중...")
        bm25_data = load_bm25()
        bm25_obj = bm25_data["bm25_obj"]
        docs = bm25_data["docs"]
        
        # 쿼리 토큰화 (split 대신 실제 분석기 사용 권장하나 테스트용으로 split)
        tokenized_query = query.split() 
        bm25_results = bm25_obj.get_top_n(tokenized_query, docs, n=3)
        
        for i, doc in enumerate(bm25_results):
            print(f"   [{i+1}] {doc.metadata.get('source')} | 내용: {doc.page_content[:50]}...")
    except Exception as e:
        print(f"   ❌ BM25 로드 실패: {e}")

if __name__ == "__main__":
    # 테스트하고 싶은 질문을 넣어보세요!
    test_query = "충북연구원 입찰 공고" 
    verify_indices(test_query)