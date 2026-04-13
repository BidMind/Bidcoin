import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 루트 디렉토리 기준 .env 로드
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# 내부 모듈 임포트
from config import DATABASE_DIR, OUTPUT_DIR
from src.parsing.concat import run_full_pipeline
from src.pipeline.rag_pipeline import meta_pipeline, chunk_pipeline
from src.embedding.vector_store import build_and_save_db


def update_entire_pipeline(use_cleaning: bool = True, use_meta_prefix: bool = True):
    print("="*60)
    print("[Bid Coin] 전체 데이터 파이프라인 업데이트 시작...")
    print("="*60)
    
    # 불러오기
    try:
        data_path = DATABASE_DIR / "data_list.csv"
        df = pd.read_csv(data_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"❌ [에러] 원본 메타데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    # [1] PDF/HWP 파싱 → 병합
    print("\n PDF 및 HWP 파싱 진행중...")
    df_parsed = run_full_pipeline(folder_path=DATABASE_DIR / "files", output_dir=OUTPUT_DIR)

    # [2] 메타데이터 정제 및 재파싱
    print("\n 메타데이터 정제 및 재파싱 진행중...")
    df_meta = meta_pipeline(df, use_cleaning=use_cleaning)

    # [3] 텍스트 청킹 및 메타데이터 결합 진행
    print("\n 텍스트 청킹 및 메타데이터 결합 진행...")
    df_chunked = chunk_pipeline(df_parsed, df_meta, use_meta_prefix=use_meta_prefix)

    # [4] FAISS 벡터 스토어 구축 및 저장
    print("\n 벡터 스토어 구축 및 저장 진행중...")
    # build_and_save_db()는 내부적으로 config.CSV_PATH에 저장된 결과를 읽어옴
    build_and_save_db()

    print("="*60)
    print("[Bid Coin] 파싱부터 DB 구축까지 모든 과정이 완료되었습니다!")
    print("이제 검색이 가능합니다.")   
    print("="*60)

if __name__ == "__main__":
    update_entire_pipeline()
    