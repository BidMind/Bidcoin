import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd

from src.parsing.concat import run_full_pipeline
from src.preprocessing.metadata_cleaning import process_metadata
from src.pipeline.rag_pipeline import meta_pipeline
from src.ingestion.chunker import process_chunking
from src.embedding.vector_store import build_and_save_db

def update_entrie_pipeline():
    print("="*50)
    print("[Bid Coin] 전체 데이터 파이프라인 업데이트 시작...")
    print("="*50)

    # 경로 설정
    SHARED_FILES_DIR = "/home/shared/files"
    OUTPUT_DIR = "/home/bidcoin"

    # [1] PDF/HWP 파싱 → 병합
    print("\n PDF 및 HWP 파싱 진행중...")
    df_parsed = run_full_pipeline(folder_path=SHARED_FILES_DIR, output_dir=OUTPUT_DIR)

    # [2] 메타데이터 정제 및 재파싱
    print("\n 메타데이터 정제 및 재파싱 진행중...")
    OPTION = True
    df_meta = meta_pipeline()

    # [3] 텍스트 청킹 및 메타데이터 결합 진행
    print("\n 텍스트 청킹 및 메타데이터 결합 진행...")
    chunked_df = process_chunking(df_parsed, df_meta)

    # [4] FAISS 벡터 스토어 구축 및 저장
    print("\n 벡터 스토어 구축 및 저장 진행중...")
    build_and_save_db()

    print("="*50)
    print("[Bid Coin] 파싱부터 DB 구축까지 모든 과정이 완료되었습니다!")
    print("이제 검색이 가능합니다.")   
    print("="*50)

if __name__ == "__main__":
    update_entrie_pipeline()
    