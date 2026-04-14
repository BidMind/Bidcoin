from __future__ import annotations
import os
import pandas as pd
import pickle # BM25 저장을 위해 추가
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi # BM25 엔진
import config
from src.embedding.embedder import get_embedding_model

def build_and_save_db():
    """
    [기능] FAISS(벡터)와 BM25(키워드) 인덱스를 동시에 생성하고 저장합니다.
    """
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {config.CSV_PATH}")

    # 1. 데이터 로드 및 Document 객체화
    df = pd.read_csv(config.CSV_PATH)
    # 피드백 반영: 메타데이터에 더 많은 정보를 미리 넣어둡니다.
    docs = [
        Document(
            page_content=row['content'], 
            metadata={
                "source": row['source'],
                "category": row.get('category', 'unknown'), # 메타데이터 확장 예시
            }
        ) for _, row in df.iterrows()
    ]
    
    # 2. [Vector] FAISS 구축 및 저장
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(config.FAISS_INDEX_DIR)
    
    # 3. [Keyword] BM25 구축 및 저장
    tokenized_corpus = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # BM25용 데이터(객체 + 원본 문서)를 pickle로 저장
    bm25_data = {
        "bm25_obj": bm25,
        "docs": docs
    }
    with open(os.path.join(config.FAISS_INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"✅ FAISS & BM25 인덱스 저장 완료: {config.FAISS_INDEX_DIR}")

def load_db():
    """FAISS를 로드합니다."""
    embeddings = get_embedding_model()
    return FAISS.load_local(config.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def load_bm25():
    """저장된 BM25 데이터를 로드합니다."""
    path = os.path.join(config.FAISS_INDEX_DIR, "bm25.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("BM25 인덱스가 존재하지 않습니다.")
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    print("🚀 BM25 인덱스 생성을 시작합니다...")
    try:
        build_and_save_db()
        print("✨ 인덱스 생성 및 저장 완료!")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")