from __future__ import annotations

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

def _safe_str(val):
    if pd.isna(val):
        return "미상"
    return str(val).strip()


def _build_content_prefix(row: pd.Series) -> str:
    return (
        f"[발주기관: {_safe_str(row.get('발주 기관', '미상'))} | "
        f"사업명: {_safe_str(row.get('사업명', '미상'))}]"
    )


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

def _chunk_pdf_row(row: pd.Series, use_meta_prefix: bool = True) -> list[dict]:
    """PDF 단일 행 clean_text 청킹."""
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

    if use_meta_prefix:
        prefix = _build_content_prefix(row)
        contents = [f"{prefix}\n\n{chunk}" for chunk in chunks]
    else:
        contents = chunks

    return [
        {
            "content": content,
            "body_length": len(chunk),
            "chunk_index": i,
            "total_chunks": total,
            "chunk_type": "text",
        }
        for i, (chunk, content) in enumerate(zip(chunks, contents))
    ]


# ============================================================
# HWP 구조형 청킹
# ============================================================

def _chunk_hwp_row(row: pd.Series, use_meta_prefix: bool = True, include_tables: bool = True,
    ) -> list[dict]:
    """
    HWP 단일 행 구조형 청킹.

    우선순위
    --------
    1) sections 있으면 섹션 단위로 청킹
    2) sections 없으면 clean_text fallback (PDF와 동일 방식)

    include_tables : True면 표 단위 청크 추가, False면 표 청킹 생략
    """
    prefix = _build_content_prefix(row) if use_meta_prefix else ""
    chunks: list[dict] = []

    sections = _parse_json_col(row.get("sections"))
    tables   = _parse_json_col(row.get("tables")) if include_tables else []

    # ------------------------------------------------------------------
    # 1) 섹션 단위 청킹
    # ------------------------------------------------------------------
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
            content = f"{prefix}\n\n{chunk}" if prefix else chunk
            chunks.append({
                "content": content,
                "body_length": len(chunk),
                "chunk_index": i,
                "total_chunks": total,
                "chunk_type": "section",
            })

    # ------------------------------------------------------------------
    # 2) sections 없으면 clean_text fallback
    # ------------------------------------------------------------------
    else:
        clean_text = str(row.get("clean_text") or "").strip()
        if clean_text:
            splitter = _get_splitter_pdf(len(clean_text))
            text_chunks = splitter.split_text(clean_text)
            total = len(text_chunks)
            for i, chunk in enumerate(text_chunks):
                content = f"{prefix}\n\n{chunk}" if prefix else chunk
                chunks.append({
                    "content": content,
                    "body_length": len(chunk),
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_type": "text",
                })

    # ------------------------------------------------------------------
    # 3) 표 단위 청크
    # ------------------------------------------------------------------
    for t in tables:
        markdown = t.get("markdown") or ""
        if not markdown.strip():
            continue

        section_title = t.get("section_title") or ""
        table_text = f"### {section_title} 표\n{markdown}" if section_title else markdown

        splitter = _get_splitter_hwp_table(len(table_text))
        table_chunks = splitter.split_text(table_text)

        for sub in table_chunks:
            content = f"{prefix}\n\n{sub}" if prefix else sub
            chunks.append({
                "content": content,
                "body_length": len(sub),
                "chunk_index": len(chunks),
                "total_chunks": None,
                "chunk_type": "table",
            })

    return chunks


# ============================================================
# 메인 함수
# ============================================================

META_COLS = ["공고 번호", "공고 차수", "발주 기관", "사업명", "사업 금액",
             "입찰 참여 시작일", "입찰 참여 마감일"]


def process_chunking_v2(
    df_pdf_parsed: pd.DataFrame,
    df_hwp_parsed: pd.DataFrame,
    df_meta: pd.DataFrame,
    use_meta_prefix: bool = True,
    include_tables: bool = True, 
) -> pd.DataFrame:
    """
    PDF/HWP 각각 청킹 후 concat.

    Parameters
    ----------
    df_pdf_parsed : PDF 파싱 결과 (csv 로드)
                    컬럼: 파일명, 파일형식, raw_text, clean_text
    df_hwp_parsed : HWP 파싱 결과 (pkl 로드)
                    컬럼: 파일명, 파일형식, raw_text, clean_text, sections, tables
    df_meta       : 메타데이터 csv
    use_meta_prefix : 청크 앞에 [발주기관|사업명] 부착 여부
    include_tables : 표 청킹 여부

    Returns
    -------
    pd.DataFrame : 청크 단위 데이터프레임
    """
    meta_cols_for_merge = ["파일명"] + [c for c in META_COLS if c in df_meta.columns]

    # ------------------------------------------------------------------
    # PDF 청킹
    # ------------------------------------------------------------------
    print(f"PDF 청킹 시작: {len(df_pdf_parsed)}건")
    df_pdf = df_pdf_parsed.copy()

    if "rag_text" in df_pdf.columns and "clean_text" not in df_pdf.columns:
        df_pdf = df_pdf.rename(columns={"rag_text": "clean_text"})
    elif "rag_text" in df_pdf.columns and "clean_text" in df_pdf.columns:
        df_pdf = df_pdf.drop(columns=["rag_text"])

    df_pdf = df_pdf.merge(df_meta[meta_cols_for_merge], on="파일명", how="left")
    df_pdf["_chunks"] = df_pdf.apply(
        lambda row: _chunk_pdf_row(row, use_meta_prefix), axis=1
    )

    chunked_pdf = df_pdf.explode("_chunks").reset_index(drop=True)
    chunked_pdf = chunked_pdf[chunked_pdf["_chunks"].notna()].copy()
    chunk_fields_pdf = pd.json_normalize(chunked_pdf["_chunks"])
    chunked_pdf = chunked_pdf.drop(columns=["_chunks"]).reset_index(drop=True)
    chunked_pdf = pd.concat([chunked_pdf, chunk_fields_pdf], axis=1)
    print(f"  PDF 청크 수: {len(chunked_pdf)}")

    # ------------------------------------------------------------------
    # HWP 청킹
    # ------------------------------------------------------------------
    print(f"HWP 청킹 시작: {len(df_hwp_parsed)}건")
    df_hwp = df_hwp_parsed.merge(df_meta[meta_cols_for_merge], on="파일명", how="left")
    df_hwp["_chunks"] = df_hwp.apply(
        lambda row: _chunk_hwp_row(row, use_meta_prefix, include_tables), axis=1
    )

    chunked_hwp = df_hwp.explode("_chunks").reset_index(drop=True)
    chunked_hwp = chunked_hwp[chunked_hwp["_chunks"].notna()].copy()
    chunk_fields_hwp = pd.json_normalize(chunked_hwp["_chunks"])
    chunked_hwp = chunked_hwp.drop(columns=["_chunks"]).reset_index(drop=True)
    chunked_hwp = pd.concat([chunked_hwp, chunk_fields_hwp], axis=1)
    print(f"  HWP 청크 수: {len(chunked_hwp)}")

    # ------------------------------------------------------------------
    # concat
    # ------------------------------------------------------------------
    chunked_df = pd.concat([chunked_pdf, chunked_hwp], ignore_index=True)

    # 불필요 컬럼 제거 및 이름 정리
    chunked_df = chunked_df.drop(
        columns=["sections", "tables", "sections_json", "tables_json"],
        errors="ignore"
    )
    chunked_df = chunked_df.rename(columns={"파일명": "source"})

    # 저장
    chunked_df.to_csv(config.CSV_PATH_V2, index=False, encoding="utf-8")
    print(f"\n청킹 완료: PDF {len(chunked_pdf)} + HWP {len(chunked_hwp)} = 총 {len(chunked_df)}개 청크")
    print(f"저장 완료: OUTPUT_DIR/{Path(config.CSV_PATH_V2).name}")
    print(chunked_df["chunk_type"].value_counts())

    return chunked_df


# 단독실행
if __name__ == "__main__":
    import time
    CHUNK_OPTION = False  # 핵심메타 부착여부
    INCLUDE_TABLES = False  # 표 청킹 생략(섹션 청킹만)
    print(f"\n--- chunker_v2 실행 (meta_prefix: {CHUNK_OPTION}, include_tables: {INCLUDE_TABLES}) ---")

    df_pdf_parsed = pd.read_csv(OUTPUT_DIR / "df_parsed_pdf_hard.csv", encoding="utf-8")
    df_hwp_parsed = pd.read_pickle(OUTPUT_DIR / "df_parsed_hwp_v2.pkl")
    df_meta = pd.read_csv(OUTPUT_DIR / "data_list_metadata.csv", encoding="utf-8")

    if CHUNK_OPTION:
        print("핵심 메타를 청크에 부착합니다")
    else:
        print("핵심 메타 부착을 건너뜁니다")

    if INCLUDE_TABLES:
        print("표 청킹을 포함합니다")
    else:
        print("표 청킹을 생략합니다")

    start = time.time()
    result = process_chunking_v2(
        df_pdf_parsed=df_pdf_parsed,
        df_hwp_parsed=df_hwp_parsed,
        df_meta=df_meta,
        use_meta_prefix=CHUNK_OPTION,
        include_tables=INCLUDE_TABLES,
    )
    elapsed = time.time() - start
    print(f"\n총 소요 시간: {elapsed:.2f}s")
    print(f"청크당 평균 처리 시간: {elapsed / len(result):.4f}s")