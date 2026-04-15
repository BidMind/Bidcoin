from __future__ import annotations

import os
import pickle
from pathlib import Path
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

import config
from src.embedding.embedder import get_embedding_model

def build_and_save_db():
    """
    [기능] 텍스트 데이터를 벡터(FAISS)와 키워드(BM25) 인덱스로 변환하여 영구 저장합니다.
    """
    # 원본 데이터 존재 여부 확인
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {config.CSV_PATH}")

    # CSV 데이터를 Document 객체 리스트로 변환
    # 'source' 메타데이터를 꼬리표로 달아줌
    df = pd.read_csv(config.CSV_PATH)

    METADATA_COLS = ["source", "발주 기관", "사업명", "사업 금액", "입찰 참여 마감일"]
    docs = [
        Document(
            page_content=row["content"],
            metadata={col: row.get(col, "") for col in METADATA_COLS if col in df.columns}
        )
        for _, row in df.iterrows()
    ]
    
    # ---------------------------------------------------------
    # 1. [Vector] FAISS 인덱스 구축 및 저장
    # ---------------------------------------------------------
    print("⏳ FAISS 인덱스(의미 검색) 구축 중...")
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(docs, embeddings)

    # 구축된 DB를 로컬(faiss_index)에 저장, 이후 재사용 가능
    vector_store.save_local(config.FAISS_INDEX_DIR)
    
    # ---------------------------------------------------------
    # 2. [Keyword] 한국어 토크나이저(Kiwi) 설정
    # ---------------------------------------------------------
    print("⏳ BM25 인덱스(키워드 검색) 구축 중...")
    kiwi = Kiwi()
    # 의미 없는 조사 및 기호 필터링
    STOP_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 
                 'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}

    def korean_tokenizer(text: str) -> list[str]:
        try:
            tokens = kiwi.tokenize(str(text))
            return [t.form for t in tokens if t.tag not in STOP_TAGS and len(t.form) > 1]
        except Exception:
            return str(text).split()
    
    # ---------------------------------------------------------
    # 3. [Keyword] BM25 인덱스 구축 및 저장
    # ---------------------------------------------------------
    tokenized_corpus = [korean_tokenizer(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25_obj": bm25,
        "docs": docs
    }
    
    bm25_path = os.path.join(config.FAISS_INDEX_DIR, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"✅ 하이브리드 검색 인덱스(FAISS + BM25) 저장 완료: ROOT_DIR/{Path(config.FAISS_INDEX_DIR).name}")


def load_db():
    """
    [기능] 저장된 벡터(FAISS) 데이터베이스를 불러옵니다.
    """
    if not os.path.exists(config.FAISS_INDEX_DIR):
        raise FileNotFoundError(f"FAISS 인덱스가 존재하지 않습니다: ROOT_DIR/{Path(config.FAISS_INDEX_DIR).name}")

    embeddings = get_embedding_model()
    return FAISS.load_local(config.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def load_bm25():
    """
    [기능] 저장된 키워드(BM25) 데이터베이스와 원본 문서를 불러옵니다.
    """
    bm25_path = os.path.join(config.FAISS_INDEX_DIR, "bm25.pkl")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 인덱스가 존재하지 않습니다: {bm25_path}")
    with open(bm25_path, "rb") as f:
        return pickle.load(f)
    
    
if __name__ == "__main__":
    build_and_save_db()
