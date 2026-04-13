import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd

from src.parsing.concat import run_full_pipeline
from src.pipeline.rag_pipeline import meta_pipeline, chunk_pipeline
from src.embedding.vector_store import build_and_save_db

from pathlib import Path
from dotenv import load_dotenv
from config import DATABASE_DIR, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

def update_entrie_pipeline(use_cleaning: bool = True, use_meta_prefix: bool = True):
    print("="*50)
    print("[Bid Coin] 전체 데이터 파이프라인 업데이트 시작...")
    print("="*50)
    
    # 불러오기
    df = pd.read_csv(DATABASE_DIR / "data_list.csv", encoding="utf-8")

    # [1] PDF/HWP 파싱 → 병합
    print("\n PDF 및 HWP 파싱 진행중...")
    df_parsed = run_full_pipeline(folder_path=DATABASE_DIR / "files", output_dir=OUTPUT_DIR)

    # [2] 메타데이터 정제 및 재파싱
    print("\n 메타데이터 정제 및 재파싱 진행중...")
    df_meta = meta_pipeline(use_cleaning=use_cleaning)

    # [3] 텍스트 청킹 및 메타데이터 결합 진행
    print("\n 텍스트 청킹 및 메타데이터 결합 진행...")
    df_chunked = chunk_pipeline(df_parsed, df_meta, use_meta_prefix=use_meta_prefix)

    # [4] FAISS 벡터 스토어 구축 및 저장
    print("\n 벡터 스토어 구축 및 저장 진행중...")
    build_and_save_db()

    print("="*50)
    print("[Bid Coin] 파싱부터 DB 구축까지 모든 과정이 완료되었습니다!")
    print("이제 검색이 가능합니다.")   
    print("="*50)

if __name__ == "__main__":
    update_entrie_pipeline()
    