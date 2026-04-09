from __future__ import annotations

import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config
from src.embedding.embedder import get_embedding_model

def build_and_save_db():
    """
    [기능] 텍스트 데이터를 벡터로 변환하여 물리적인 폴더에 영구 저장합니다.
    """
    # 원본 데이터 존재 여부 확인
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {config.CSV_PATH}")

    # CSV 데이터를 Document 객체 리스트로 변환
    # 'source' 메타데이터를 꼬리표로 달아줌
    df = pd.read_csv(config.CSV_PATH)
    docs = [Document(page_content=row['content'], metadata={"source": row['source']}) for _, row in df.iterrows()]
    
    # 임베딩 모델을 가져와서 텍스트를 숫자로 변환 후 FAISS DB 구축
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(docs, embeddings)

    # 구축된 DB를 로컬(faiss_index)에 저장, 이후 재사용 가능
    vector_store.save_local(config.FAISS_INDEX_DIR)
    print(f"FAISS 인덱스가 성공적으로 저장되었습니다: {config.FAISS_INDEX_DIR}")

def load_db():
    """
    [기능] 1차 검색을 위해 저장된 벡터 데이터베이스를 불러옵니다.
    """
    if not os.path.exists(config.FAISS_INDEX_DIR):
        raise FileNotFoundError(f"FAISS 인덱스가 존재하지 않습니다: {config.FAISS_INDEX_DIR}")

    embeddings = get_embedding_model()
    return FAISS.load_local(config.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    build_and_save_db()
