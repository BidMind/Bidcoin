from __future__ import annotations

from pathlib import Path
import time
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config
from src.embedding.embedder import get_embedding_model

def build_and_save_db_v2(df_chunked: pd.DataFrame):
    """
    [기능] 청킹된 DataFrame을 벡터로 변환하여 FAISS 인덱스를 저장합니다.
    """
    if df_chunked is None or df_chunked.empty:
        raise ValueError("청킹된 데이터가 비어 있습니다.")
    
    print(f"청크 수: {len(df_chunked)}")

    t0 = time.time()
    METADATA_COLS = ["source", "발주 기관", "사업명", "사업 금액", "입찰 참여 마감일"]

    docs = [
        Document(
            page_content=row["content"],
            metadata={col: row.get(col, "") for col in METADATA_COLS if col in df_chunked.columns}
        )
        for _, row in df_chunked.iterrows()
    ]
    print(f"Document 변환 완료: {len(docs)}개, {time.time() - t0:.2f}s")
    
    # 임베딩 모델을 가져와서 텍스트를 숫자로 변환 후 FAISS DB 구축
    t1 = time.time()
    embeddings = get_embedding_model()
    print(f"임베딩 모델 로드 완료: {time.time() - t1:.2f}s")

    t2 = time.time()
    vector_store = FAISS.from_documents(docs, embeddings)
    print(f"FAISS 구축 완료: {time.time() - t2:.2f}s")

    # 구축된 DB를 로컬(faiss_index)에 저장, 이후 재사용 가능
    t3 = time.time()
    vector_store.save_local(config.FAISS_INDEX_DIR_V2)
    print(f"FAISS 인덱스가 성공적으로 저장되었습니다: ROOT_DIR/{Path(config.FAISS_INDEX_DIR_V2).name}")
    print(f"FAISS 저장 완료: {time.time() - t3:.2f}s")


def load_db_v2():
    """
    [기능] 1차 검색을 위해 저장된 벡터 데이터베이스를 불러옵니다.
    """
    if not os.path.exists(config.FAISS_INDEX_DIR_V2):
        raise FileNotFoundError(f"FAISS 인덱스가 존재하지 않습니다: ROOT_DIR/{Path(config.FAISS_INDEX_DIR_V2).name}")

    embeddings = get_embedding_model()
    return FAISS.load_local(config.FAISS_INDEX_DIR_V2, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    print("캐시 pkl을 직접 읽어 FAISS 인덱스만 생성하는 테스트용입니다.")
    print("즉, ATTACH_OPTION 실험이 반영되는 메인 실행 방식은 아닙니다.")
    print("전체 실험은 update_entire_pipeline_v2()를 통해 실행하세요.")
    INCLUDE_TABLES = False  # 표 청킹 포함 여부
    pkl_path = config.PKL_PATH_V21 if INCLUDE_TABLES else config.PKL_PATH_V22
    df = pd.read_pickle(pkl_path)
    build_and_save_db_v2(df)
