# src/ingestion/chunking.py

from __future__ import annotations

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


# ============================================================
# 청크 크기 결정 (clean_text 길이 기준)
# ============================================================

def _get_splitter(clean_text_len: int) -> RecursiveCharacterTextSplitter:
    """clean_text 길이에 따라 적절한 splitter 반환."""
    if clean_text_len < 500:
        chunk_size, chunk_overlap = 500, 0
    elif clean_text_len < 3000:
        chunk_size, chunk_overlap = 512, 50
    elif clean_text_len < 10000:
        chunk_size, chunk_overlap = 512, 100
    else:
        chunk_size, chunk_overlap = 1024, 200

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ============================================================
# 메타 헤더 생성 (메타데이터 df에서 가져온 보조 정보)
# ============================================================

def _safe_str(val) -> str:
    """NaN → '미상'."""
    return "미상" if pd.isna(val) else str(val)


def _format_amount(val) -> str:
    try:
        return f"{float(val):,.0f}원" if not pd.isna(val) else "미상"
    except Exception:
        return _safe_str(val)


def _build_meta_header(meta_row: pd.Series, chunk_idx: int, total_chunks: int) -> str:
    """
    메타데이터 행에서 보조 정보를 추출해 헤더 생성.
    null 허용 컬럼: 공고 번호, 공고 차수, 입찰 참여 마감일 → '미상' 처리.
    """
    return (
        f"[공고번호: {_safe_str(meta_row.get('공고 번호', '미상'))} | "
        f"공고차수: {_safe_str(meta_row.get('공고 차수', '미상'))} | "
        f"발주기관: {_safe_str(meta_row.get('발주 기관', '미상'))} | "
        f"사업명: {_safe_str(meta_row.get('사업명', '미상'))} | "
        f"금액: {_format_amount(meta_row.get('사업 금액'))} | "
        f"시작일: {_safe_str(meta_row.get('입찰 참여 시작일', '미상'))} | "
        f"마감일: {_safe_str(meta_row.get('입찰 참여 마감일', '미상'))} | "
        f"조각순서: {chunk_idx + 1}/{total_chunks}]"
    )


# ============================================================
# 행 단위 청킹 (청킹 대상: clean_text)
# ============================================================

def _chunk_row(row: pd.Series) -> list[str]:
    """
    단일 행의 clean_text를 청킹하고 메타 헤더를 붙여 반환.
    clean_text가 없으면 빈 리스트 반환.
    """
    clean_text = row.get("clean_text", "")
    if pd.isna(clean_text) or len(str(clean_text).strip()) == 0:
        return []

    clean_text = str(clean_text).strip()
    splitter = _get_splitter(len(clean_text))  # clean_text 길이 기준
    chunks = splitter.split_text(clean_text)

    return [
        f"{_build_meta_header(row, i, len(chunks))}\n\n{chunk}"
        for i, chunk in enumerate(chunks)
    ]


# ============================================================
# 메인 함수
# ============================================================

def process_chunking(
    df_parsed: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    파싱된 RFP 텍스트를 청킹하고 메타데이터를 보조로 주입.

    Parameters
    ----------
    df_parsed : pd.DataFrame
        df_parsed.csv 로드한 것. 컬럼: 파일명, 파일형식, raw_text, clean_text
    df_meta : pd.DataFrame
        data_list_metadata.csv 로드한 것. 파일명 기준으로 join.

    Returns
    -------
    pd.DataFrame
        청크 단위로 분리된 데이터프레임.
        컬럼: 파일명 + 메타 컬럼들 + 청크_텍스트 + 청크_길이
    """
    print("청킹 시작...")

    df = df_parsed.merge(df_meta, on="파일명", how="left")
    df["청크_리스트"] = df.apply(_chunk_row, axis=1)

    chunked_df = df.explode("청크_리스트").reset_index(drop=True)
    chunked_df = chunked_df.rename(columns={"청크_리스트": "청크_텍스트"})
    chunked_df = chunked_df.drop(columns=["raw_text", "clean_text"], errors="ignore")
    chunked_df["청크_길이"] = chunked_df["청크_텍스트"].apply(
        lambda x: len(x) if isinstance(x, str) else 0
    )

    # 임베딩 코드(embedder.py)가 기대하는 컬럼명으로 변환
    chunked_df = chunked_df.rename(columns={
        "청크_텍스트": "content",
        "파일명": "source",
    })

    # config.CSV_PATH에 저장 
    # (CSV_PATH = "/home/bidcoin/preprocessed_data.csv" )
    chunked_df.to_csv(config.CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"청킹 완료: 원본 {len(df_parsed)}건 → {len(chunked_df)}개 청크")
    print(f"저장 완료: {config.CSV_PATH}")

    return chunked_df

# main에서 아래 코드 필요
# ex.

# 1)파싱데이터
# df_parsed = pd.read_csv("df_parsed.csv", encoding="utf-8") 

# 2)메타데이터
# df = pd.read_csv("data_list.csv", encoding="utf-8")
# df_meta = process_metadata(df, files_dir="/home/shared/files")
# df_meta.to_csv("data_list_metadata.csv", index=False, encoding="utf-8")

# 3)청킹데이터
# process_chunking(df_parsed, df_meta)  <- 저장까지 내부에서 처리
