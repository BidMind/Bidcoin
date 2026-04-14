from __future__ import annotations

import time
import json
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

from pathlib import Path
from dotenv import load_dotenv
from config import OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


# ============================================================
# 청크 크기 결정 (clean_text 길이 기준)
# ============================================================

def _get_splitter_pdf(text_len: int) -> RecursiveCharacterTextSplitter:
    """PDF용 — 문서 전체 길이 기준"""
    if text_len < 30000:
        chunk_size, chunk_overlap = 1000, 100
    elif text_len < 80000:
        chunk_size, chunk_overlap = 1200, 150
    else:
        chunk_size, chunk_overlap = 1500, 200

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_splitter_hwp_section(text_len: int) -> RecursiveCharacterTextSplitter:
    """HWP 섹션용 — 섹션 단위 길이 기준
    - 중앙값 409, 90%가 4857 이하
    - 대부분 1청크로 들어가고 긴 것만 분할
    """
    if text_len < 1000:
        chunk_size, chunk_overlap = 800, 80
    elif text_len < 5000:
        chunk_size, chunk_overlap = 1000, 100
    else:
        chunk_size, chunk_overlap = 1200, 150

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_splitter_hwp_table(text_len: int) -> RecursiveCharacterTextSplitter:
    """HWP 표용 — 표 단위 길이 기준
    - 중앙값 431, 90%가 1220 이하
    - overlap 0 (표 구조는 이어붙이면 의미 손실)
    """
    if text_len < 1500:
        chunk_size, chunk_overlap = 1000, 0
    else:
        chunk_size, chunk_overlap = 1500, 0

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""],
    )

# ============================================================
# 헬퍼
# ============================================================


def _parse_json_col(val) -> list:
    """sections_json / tables_json 컬럼 파싱."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


# ============================================================
# PDF 청킹 (기존 방식 유지)
# ============================================================

def _chunk_pdf_row(row: pd.Series) -> list[dict]:
    """PDF 단일 행 clean_text 청킹 - 순수 본문 청크만 생성"""
    clean_text = row.get("clean_text", "")
    try:
        if not clean_text or not str(clean_text).strip():
            return []
    except Exception:
        return []

    clean_text = str(clean_text).strip()
    splitter = _get_splitter_pdf(len(clean_text))
    chunks = splitter.split_text(clean_text)
    total = len(chunks)

    return [
        {
            "content": chunk,
            "body_length": len(chunk),
            "chunk_index": i,
            "total_chunks": total,
            "chunk_type": "text",
        }
        for i, chunk in enumerate(chunks)
    ]


# ============================================================
# HWP 구조형 청킹
# ============================================================

def _chunk_hwp_row(row: pd.Series, include_tables: bool = True,
    ) -> list[dict]:
    """
    HWP 단일 행 구조형 청킹 - 순수 본문 청크만 생성

    우선순위
    --------
    1) sections 있으면 섹션 단위로 청킹
    2) sections 없으면 clean_text fallback (PDF와 동일 방식)

    include_tables : True면 표 단위 청크 추가, False면 표 청킹 생략
    """
    chunks: list[dict] = []

    sections = _parse_json_col(row.get("sections"))
    tables   = _parse_json_col(row.get("tables")) if include_tables else []

    # 1) 섹션 단위 청킹
    if sections:
        section_chunks: list[str] = []

        for sec in sections:
            title   = sec.get("section_title") or ""
            content = sec.get("content") or ""

            if not content.strip():
                continue

            sec_text = f"## {title}\n{content}" if title else content
            splitter = _get_splitter_hwp_section(len(sec_text))
            sub_chunks = splitter.split_text(sec_text)
            section_chunks.extend(sub_chunks)

        total = len(section_chunks)
        for i, chunk in enumerate(section_chunks):
            chunks.append({
                "content": chunk,      # 메타 없는 순수 본문
                "body_length": len(chunk),
                "chunk_index": i,
                "total_chunks": total,
                "chunk_type": "section",
            })

    # 2) sections 없으면 clean_text fallback
    else:
        clean_text = str(row.get("clean_text") or "").strip()
        if clean_text:
            splitter = _get_splitter_pdf(len(clean_text))
            text_chunks = splitter.split_text(clean_text)
            total = len(text_chunks)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk,  # 메타 없는 순수 본문
                    "body_length": len(chunk),
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_type": "text",
                })

    # 3) 표 단위 청크
    for t in tables:
        markdown = t.get("markdown") or ""
        if not markdown.strip():
            continue

        section_title = t.get("section_title") or ""
        table_text = f"### {section_title} 표\n{markdown}" if section_title else markdown

        splitter = _get_splitter_hwp_table(len(table_text))
        table_chunks = splitter.split_text(table_text)

        for sub in table_chunks:
            chunks.append({
                "content": sub,        # 메타 없는 순수 본문
                "body_length": len(sub),
                "chunk_index": len(chunks),
                "total_chunks": None,
                "chunk_type": "table",
            })

    return chunks


# ============================================================
# 메타부착 후처리
# ============================================================

# 포맷
def format_amount(val) -> str:
    try:
        amount = float(val)
        if amount >= 1_0000_0000:  # 1억 이상
            million_100 = amount / 1_0000_0000
            return f"{million_100:g}억원"
        elif amount >= 1000_0000:  # 1천만 이상
            million_10 = amount / 1000_0000
            return f"{million_10:g}천만원"
        else:
            return f"{int(amount):,}원"
    except:
        return "미상"

def format_deadline(val) -> str:
    try:
        # pandas Timestamp나 문자열 모두 처리
        dt = pd.to_datetime(str(val), errors="coerce")
        if pd.isna(dt):
            return "미상"
        return f"{dt.year}년 {dt.month}월 {dt.day}일"
    except:
        return "미상"

def attach_meta_prefix(df_chunked: pd.DataFrame) -> pd.DataFrame:
    """
    순수 본문 청크에 핵심 메타 prefix를 후처리로 부착
    """
    df = df_chunked.copy()

    org = df["발주 기관"].fillna("미상").astype(str).str.strip()
    biz = df["사업명"].fillna("미상").astype(str).str.strip()
    amount = df["사업 금액"].apply(format_amount)
    deadline = df["입찰 참여 마감일"].apply(format_deadline)
    body = df["content"].fillna("").astype(str).str.strip()

    prefix = (
        "[발주기관: " + org + 
        " | 사업명: " + biz + 
        " | 사업금액: " + amount + 
        " | 입찰마감일: " + deadline + "]"
    )
    mask = body.ne("")
    df.loc[mask, "content"] = prefix[mask] + "\n\n" + body[mask]
    df.loc[~mask, "content"] = body[~mask]

    return df


# ============================================================
# 메인 함수
# ============================================================

META_COLS = ["공고 번호", "공고 차수", "발주 기관", "사업명", "사업 금액",
             "입찰 참여 시작일", "입찰 참여 마감일"]


def process_chunking_v2(
    df_meta: pd.DataFrame,
    use_meta_prefix: bool = True,
    include_tables: bool = True, 
    use_cache: bool = False
) -> pd.DataFrame:
    """
    PDF/HWP 각각 청킹 후 concat.

    Parameters
    ----------
    df_meta         : 메타데이터 csv
    use_meta_prefix : 핵심메타 부착여부
    include_tables  : 표 청킹 여부
    use_cache       : True면 캐시 파일 존재 시 청킹 건너뛰고 로드

    Returns
    -------
    pd.DataFrame : 청크 단위 데이터프레임
    """

    # 저장 경로 분기
    if include_tables:
        cache_path = config.PKL_PATH_V21
    else:
        cache_path = config.PKL_PATH_V22

    meta_cols_for_merge = ["파일명"] + [c for c in META_COLS if c in df_meta.columns]

    # 1) 캐시 로드
    if use_cache and Path(cache_path).exists():
        print(f"캐시 로드: OUTPUT_DIR/{Path(cache_path).name}")
        t_cache = time.time()
        df_chunked = pd.read_pickle(cache_path)
        print(f"캐시 pkl 로드 시간: {time.time() - t_cache:.2f}s")

    else: 
        df_pdf_parsed = pd.read_csv(OUTPUT_DIR / "df_parsed_pdf_hard.csv", encoding="utf-8")
        df_hwp_parsed = pd.read_pickle(OUTPUT_DIR / "df_parsed_hwp_v2.pkl")

    # 2) PDF 청킹
        print(f"PDF 청킹 시작: {len(df_pdf_parsed)}건")
        df_pdf = df_pdf_parsed.copy()

        if "rag_text" in df_pdf.columns and "clean_text" not in df_pdf.columns:
            df_pdf = df_pdf.rename(columns={"rag_text": "clean_text"})
        elif "rag_text" in df_pdf.columns and "clean_text" in df_pdf.columns:
            df_pdf = df_pdf.drop(columns=["rag_text"])

        df_pdf = df_pdf.merge(df_meta[meta_cols_for_merge], on="파일명", how="left")
        df_pdf["_chunks"] = df_pdf.apply(
            lambda row: _chunk_pdf_row(row), axis=1
        )

        chunked_pdf = df_pdf.explode("_chunks").reset_index(drop=True)
        chunked_pdf = chunked_pdf[chunked_pdf["_chunks"].notna()].copy()
        chunk_fields_pdf = pd.json_normalize(chunked_pdf["_chunks"])
        chunked_pdf = chunked_pdf.drop(columns=["_chunks"]).reset_index(drop=True)
        chunked_pdf = pd.concat([chunked_pdf, chunk_fields_pdf], axis=1)
        print(f"  PDF 청크 수: {len(chunked_pdf)}")

        # 3) HWP 청킹
        print(f"HWP 청킹 시작: {len(df_hwp_parsed)}건")
        df_hwp = df_hwp_parsed.merge(df_meta[meta_cols_for_merge], on="파일명", how="left")
        df_hwp["_chunks"] = df_hwp.apply(
            lambda row: _chunk_hwp_row(row, include_tables), axis=1
        )

        chunked_hwp = df_hwp.explode("_chunks").reset_index(drop=True)
        chunked_hwp = chunked_hwp[chunked_hwp["_chunks"].notna()].copy()
        chunk_fields_hwp = pd.json_normalize(chunked_hwp["_chunks"])
        chunked_hwp = chunked_hwp.drop(columns=["_chunks"]).reset_index(drop=True)
        chunked_hwp = pd.concat([chunked_hwp, chunk_fields_hwp], axis=1)
        print(f"  HWP 청크 수: {len(chunked_hwp)}")

        # 3) concat
        chunked_df = pd.concat([chunked_pdf, chunked_hwp], ignore_index=True)

        # 불필요 컬럼 제거 및 이름 정리
        chunked_df = chunked_df.drop(
            columns=["sections", "tables", "sections_json", "tables_json"],
            errors="ignore"
        )
        chunked_df = chunked_df.rename(columns={"파일명": "source"})

        # 4) 캐시저장
        chunked_df.to_pickle(cache_path)
        print(f"\n청킹 완료: PDF {len(chunked_pdf)} + HWP {len(chunked_hwp)} = 총 {len(chunked_df)}개 청크")
        print(f"저장 완료: OUTPUT_DIR/{Path(cache_path).name}")
        print(chunked_df["chunk_type"].value_counts())

        df_chunked = chunked_df

    # 5) 메타 부착 후처리
    if use_meta_prefix:
        print("핵심 메타를 청크에 부착합니다")
        df_chunked = attach_meta_prefix(df_chunked)
    else:
        print("핵심 메타 부착을 건너뜁니다")

    return df_chunked


# 단독실행 (T/F 바꿔보기) 
# 1회차 캐시생성시 False/True/False & False/False/False, 이후 캐시로드시 USE_CACHE=True
if __name__ == "__main__":
    import time
    ATTACH_OPTION   = False   # 핵심메타 부착여부
    INCLUDE_TABLES = False  # True: 표 포함 / False: 표 제외
    USE_CACHE      = False  # 캐시 사용여부
    print(f"\n--- chunker_v2 실행 (meta_prefix: {ATTACH_OPTION}, include_tables: {INCLUDE_TABLES}, cache: {USE_CACHE}) ---")

    df_meta = pd.read_csv(OUTPUT_DIR / "data_list_metadata.csv", encoding="utf-8")

    if INCLUDE_TABLES:
        print("표 청킹을 포함합니다")
    else:
        print("표 청킹을 생략합니다")

    start = time.time()
    chunked_df = process_chunking_v2(
        df_meta=df_meta,
        use_meta_prefix=ATTACH_OPTION,
        include_tables=INCLUDE_TABLES,
        use_cache=USE_CACHE
    )
    elapsed = time.time() - start

    print(f"\n총 소요 시간: {elapsed:.2f}s")
    print(f"청크당 평균 처리 시간: {elapsed / len(chunked_df):.4f}s")