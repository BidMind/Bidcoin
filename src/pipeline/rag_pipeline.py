import pandas as pd
from src.preprocessing.metadata_cleaning import process_metadata
from src.ingestion.chunker import process_chunking
from src.parsing.concat import run_full_pipeline

from pathlib import Path
from dotenv import load_dotenv
from config import DATABASE_DIR, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


# 메타데이터 정제 옵션
def meta_pipeline(use_cleaning=True):
    print(f"\n--- start meta pipeline (cleaning: {use_cleaning}) ---")

    input_csv_path = DATABASE_DIR / "data_list.csv"
    files_dir = DATABASE_DIR / "files"
    output_csv_path = OUTPUT_DIR / "data_list_metadata.csv"
    
    # 1. 메타데이터 처리 단계(옵션이 True일 때만 정제 모듈 실행)
    if use_cleaning:  
        df = pd.read_csv(input_csv_path, encoding="utf-8")  # 메타데이터 불러오기
        df_meta = process_metadata(df, files_dir=str(files_dir))  # 정제모듈 호출
        df_meta.to_csv(output_csv_path, index=False, encoding="utf-8")  # 메타데이터 저장
        print("메타데이터 정제 완료")
        print(f"저장 완료: OUTPUT_DIR/{output_csv_path.name}\n")

    else:
        print("메타데이터 정제를 건너뜁니다")
        df_meta = pd.read_csv(input_csv_path, encoding="utf-8")
        print(f"사용 파일: DATABASE_DIR/{input_csv_path.name}\n")
    
    return df_meta


# 2. 청킹 파이프라인
def chunk_pipeline(df_parsed: pd.DataFrame, df_meta: pd.DataFrame, use_meta_prefix: bool = True
    ) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_meta : pd.DataFrame
        meta_pipeline()의 반환값을 그대로 전달
    use_meta_prefix : bool
        True  → 각 청크 content 앞에 [발주기관|사업명] 헤더 부착
        False → 본문 청크만 저장
    """
    print(f"\n--- start chunk pipeline (meta_prefix: {use_meta_prefix}) ---")
 
    if use_meta_prefix:
        print("핵심 메타를 청크에 부착합니다")
        df_chunked = process_chunking(df_parsed, df_meta, use_meta_prefix=True)
        
    else:
        print("핵심 메타 부착을 건너뜁니다")
        df_chunked = process_chunking(df_parsed, df_meta, use_meta_prefix=False)

    print("\ncontent 샘플 확인:")
    for i, text in enumerate(df_chunked["content"].head(3), start=1):
        preview = str(text).replace("\n", " ")[:120]
        print(f"[{i}] {preview}...")
 
    return df_chunked


# 작업파일 단독실행용 (여기서 T/F 파라미터 바꿔보기)
if __name__ == "__main__":
    df_meta = meta_pipeline(use_cleaning=False)
    df_parsed = run_full_pipeline(folder_path=DATABASE_DIR / "files", output_dir=OUTPUT_DIR)
    df_chunked = chunk_pipeline(df_parsed, df_meta, use_meta_prefix=True)

# run_pipe 작성 방법 <-run_pipe는 현수님이 만든 모듈 임의명칭
# def run_pipe(use_cleaning: bool = True, use_meta_prefix: bool = True):
#     df_meta = meta_pipeline(use_cleaning=use_cleaning)   # 1. 메타데이터 정제
#     df_parsed = run_full_pipeline(folder_path=DATABASE_DIR / "files", output_dir=OUTPUT_DIR)  # 2.df_parsed.csv생성
#     df_chunked = chunk_pipeline(df_parsed, df_meta, use_meta_prefix=use_meta_prefix)  # 3. 청킹
#     return df_chunked

# ----- main.py 에 필요한 코드 -----
# from src.pipeline.rag_pipeline import run_pipe 
#
# if __name__ == "__main__":
#     META_OPTION = True  # 옵션값을 여기서 조절
#     CHUNK_OPTION = True
#     run_pipe(use_cleaning=META_OPTION, use_meta_prefix=CHUNK_OPTION)
