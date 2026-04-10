# src/ingestion/chunker.py

from __future__ import annotations

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from src.preprocessing.metadata_cleaning import process_metadata

from pathlib import Path
from dotenv import load_dotenv
from config import DATABASE_DIR, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parent  
load_dotenv(ROOT_DIR / ".env")


# ============================================================
# 청크 크기 결정 (clean_text 길이 기준)
# ============================================================

def _get_splitter(clean_text_len: int) -> RecursiveCharacterTextSplitter:
    """clean_text 길이에 따라 적절한 splitter 반환."""
    if clean_text_len < 30000:
        chunk_size, chunk_overlap = 1000, 100
    elif clean_text_len < 80000:
        chunk_size, chunk_overlap = 1200, 150
    else:
        chunk_size, chunk_overlap = 1500, 200


    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ============================================================
# 헬퍼
# ============================================================

def _safe_str(val):
    if pd.isna(val):
        return "미상"
    return str(val).strip()
    
    
# ============================================================
# 메타 헤더 생성 (메타데이터 df에서 가져온 보조 정보)
# ============================================================

def _build_content_prefix(row: pd.Series) -> str:
    """
    임베딩 content 앞에 붙이는 핵심 메타 2개.
    리트리벌 쿼리와 직접 매칭되는 검색 키 역할.
    """
    return (
        f"[발주기관: {_safe_str(row.get('발주 기관', '미상'))} | "
        f"사업명: {_safe_str(row.get('사업명', '미상'))}]"
    )


# ============================================================
# 행 단위 청킹 (청킹 대상: clean_text)
# ============================================================

def _chunk_row(row: pd.Series) -> list[dict]:
    """
    단일 행의 clean_text를 청킹.
    반환값: 청크별 dict 리스트.
      - content     : [핵심메타]\n\n{본문청크}  (임베딩 대상)
      - body_length : 본문 청크 길이 (헤더 제외, 청킹 품질 확인용)
      - chunk_index : 0-based 청크 순서
      - total_chunks: 해당 문서의 전체 청크 수
    """
    clean_text = row.get("clean_text", "")
    if pd.isna(clean_text) or len(str(clean_text).strip()) == 0:
        return []

    clean_text = str(clean_text).strip()
    splitter = _get_splitter(len(clean_text))  # clean_text 길이 기준
    chunks = splitter.split_text(clean_text)

    prefix = _build_content_prefix(row)
    total = len(chunks)

    return [
        {
            "content": f"{prefix}\n\n{chunk}",
            "body_length": len(chunk),
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk in enumerate(chunks)
    ]


# ============================================================
# 메인 함수
# ============================================================

META_COLS = ["공고 번호", "공고 차수", "발주 기관", "사업명", "사업 금액",
             "입찰 참여 시작일", "입찰 참여 마감일"]

def process_chunking(
    df_parsed: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    파싱된 RFP 텍스트를 청킹하고 메타데이터를 병합.
 
    Parameters
    ----------
    df_parsed : pd.DataFrame
        df_parsed.csv 로드한 것. 컬럼: 파일명, 파일형식, raw_text, clean_text
    df_meta : pd.DataFrame
        data_list_metadata.csv 로드한 것. 파일명 기준으로 join.
 
    Returns
    -------
    pd.DataFrame
        청크 단위 데이터프레임. 주요 컬럼:
          source        : 원본 파일명
          content       : [발주기관|사업명] + 본문청크  (임베딩 대상)
          body_length   : 본문 청크 길이 (헤더 제외)
          chunk_index   : 청크 순서 (0-based)
          total_chunks  : 문서 내 전체 청크 수
          공고 번호 / 공고 차수 / 발주 기관 / 사업명 /
          사업 금액 / 입찰 참여 시작일 / 입찰 참여 마감일  : 별도 메타 컬럼
    """
    print("청킹 시작...")

    # df_meta에서 필요한 컬럼만 골라 merge
    meta_cols_for_merge = ["파일명"] + [c for c in META_COLS if c in df_meta.columns]
    df = df_parsed.merge(df_meta[meta_cols_for_merge], on="파일명", how="left")
 
    # 청킹: 각 행 → dict 리스트
    df["_chunks"] = df.apply(_chunk_row, axis=1)
 
    chunked_df = df.explode("_chunks").reset_index(drop=True)
 
    # explode 후 NaN 행 제거 (clean_text가 비어 있던 행)
    chunked_df = chunked_df[chunked_df["_chunks"].notna()].copy()
 
    # dict 컬럼 펼치기
    chunk_fields = pd.json_normalize(chunked_df["_chunks"])
    chunked_df = chunked_df.drop(columns=["_chunks"]).reset_index(drop=True)
    chunked_df = pd.concat([chunked_df, chunk_fields], axis=1)
 
    # 불필요 컬럼 제거 및 이름 정리
    chunked_df = chunked_df.drop(columns=["raw_text", "clean_text"], errors="ignore")
    chunked_df = chunked_df.rename(columns={"파일명": "source"})
 
    # config.CSV_PATH에 저장
    chunked_df.to_csv(config.CSV_PATH, index=False, encoding="utf-8")
    print(f"청킹 완료: 원본 {len(df_parsed)}건 → {len(chunked_df)}개 청크")
    print(f"저장 완료: {config.CSV_PATH}")
 
    return chunked_df

if __name__ == "__main__":
    df_parsed = pd.read_csv(OUTPUT_DIR / "df_parsed.csv", encoding="utf-8")
    df_meta = process_metadata(files_dir= DATABASE_DIR / "files")
    process_chunking(df_parsed, df_meta)

# main에서 아래 코드 필요
# ex.

# 1)파싱데이터
# df_parsed = pd.read_csv("/home/bidcoin/df_parsed.csv", encoding="utf-8") 

# 2)메타데이터
# - 모듈내부에서 /home/shared/data_list.csv 를 불러옴
# - 메타데이터 처리 후 /home/bidcoin/data_list_metadata.csv 로 저장
# df_meta = process_metadata()

# 3)청킹데이터
# process_chunking(df_parsed, df_meta)  <- 저장까지 내부에서 처리
