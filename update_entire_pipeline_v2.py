import os
import sys
import time
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
from config import DATABASE_DIR
from src.pipeline.rag_pipeline import meta_pipeline, chunk_pipeline_v2 
from src.embedding.vector_store_v2 import build_and_save_db_v2


def update_entire_pipeline_v2(
        use_cleaning: bool = True, 
        use_meta_prefix: bool = True,
        include_tables: bool = True,
        use_cache: bool = True):
    print("="*60)
    print("[Bid Coin] 전체 데이터 파이프라인 업데이트 시작...")
    print("="*60)
    total_start = time.time()
    
    # [1] 불러오기
    try:
        data_path = DATABASE_DIR / "data_list.csv"
        df = pd.read_csv(data_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"[에러] 원본 메타데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    # [2] 메타데이터 정제 및 재파싱
    print("\n 메타데이터 정제 및 재파싱 진행중...")
    t1 = time.time()
    df_meta = meta_pipeline(df, use_cleaning=use_cleaning)
    print(f" 메타데이터 정제 완료: {time.time() - t1:.2f}s")

    # [3] 텍스트 청킹 및 메타데이터 결합 진행
    print("\n 텍스트 청킹 및 메타데이터 결합 진행...")
    t2 = time.time()
    df_chunked = chunk_pipeline_v2(df_meta, use_meta_prefix=use_meta_prefix, include_tables=include_tables, use_cache=use_cache)
    print(f" 청킹 완료: {time.time() - t2:.2f}s")
    
    # [4] FAISS 벡터 스토어 구축 및 저장
    print("\n 벡터 스토어 구축 및 저장 진행중...")
    t3 = time.time()
    build_and_save_db_v2(df_chunked)
    print(f" 벡터 스토어 완료: {time.time() - t3:.2f}s")
    print(f"총 소요 시간: {time.time() - total_start:.2f}s")

    print("="*60)
    print("[Bid Coin] 파싱부터 DB 구축까지 모든 과정이 완료되었습니다!")
    print("이제 검색이 가능합니다.")   
    print("="*60)

if __name__ == "__main__":
    META_OPTION = True     # 메타데이터 정제 여부
    ATTACH_OPTION = True   # 청크에 핵심메타 부착여부
    INCLUDE_TABLES = False  # 표 청킹 포함 여부
    USE_CACHE = True       # 캐시 사용 여부
    update_entire_pipeline_v2(use_cleaning=META_OPTION, use_meta_prefix=ATTACH_OPTION, include_tables=INCLUDE_TABLES, use_cache=USE_CACHE)