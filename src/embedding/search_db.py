import os
from langchain_community.vectorstores import FAISS
from embedding import get_embedding_model  # 분리한 임베딩 모듈 호출

# 📍 경로 설정 (실제 최종 경로)
BASE_DIR = r"D:\VS Test\bidcoin" # 수정
FAISS_DB_PATH = os.path.join(BASE_DIR, "faiss_index")

def search_from_saved_db(query: str, k: int = 15):
    if not os.path.exists(FAISS_DB_PATH):
        print("❌ 저장된 FAISS DB가 없습니다. build_db.py를 먼저 실행하세요.")
        return []

    print(f"\n🔍 [검색 모드] 로컬 DB에서 질문을 검색합니다: '{query}'")
    
    # 1. 전용 모듈에서 임베딩 모델 가져오기
    embeddings = get_embedding_model()
    
    # 2. 로컬에 저장된 FAISS DB 불러오기
    # allow_dangerous_deserialization=True 는 로컬에서 내가 만든 안전한 DB를 부를 때 필수 옵션입니다.
    vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 3. 검색 수행
    candidates = vectorstore.similarity_search(query, k=k)
    print(f"상위 {len(candidates)}개 문서 검색 완료.")
    
    return candidates

if __name__ == "__main__":
    q = "입찰 참여를 위한 필수 자격 요건은 무엇인가요?"
    results = search_from_saved_db(q, k=3)
    
    for i, doc in enumerate(results):
        print(f"\n[{i+1}위] 출처: {doc.metadata.get('source', '알수없음')}")
        print(f"내용: {doc.page_content[:150]}...")