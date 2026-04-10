import pandas as pd
from src.preprocessing.metadata_cleaning import process_metadata

from pathlib import Path
from dotenv import load_dotenv
from config import DATABASE_DIR, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


# 메타데이터 정제 옵션
def meta_pipeline(use_cleaning=True):
    print(f"--- start meta pipeline (cleaning: {use_cleaning}) ---")

    input_csv_path = DATABASE_DIR / "data_list.csv"
    files_dir = DATABASE_DIR / "files"
    output_csv_path = OUTPUT_DIR / "data_list_metadata.csv"
    
    # 1. 메타데이터 처리 단계(옵션이 True일 때만 정제 모듈 실행)
    if use_cleaning:  
        df = pd.read_csv(input_csv_path, encoding="utf-8")  # 메타데이터 불러오기
        df_meta = process_metadata(df, files_dir=str(files_dir))  # 정제모듈 호출
        df_meta.to_csv(output_csv_path, index=False, encoding="utf-8")  # 메타데이터 저장
        print("메타데이터 정제 완료")
        print(f"저장 완료: {output_csv_path}")
    else:
        print("메타데이터 정제를 건너뜁니다")
        df_meta = pd.read_csv(input_csv_path, encoding="utf-8")
    
    return df_meta


# 작업파일 단독실행용
if __name__ == "__main__":
    meta_pipeline(use_cleaning=True)


# 2. 파싱 및 청킹 단계 (추후 구현)
    # ...