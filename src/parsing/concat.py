"""
concat.py — PDF·HWP 파싱 결과 통합 모듈

주요 함수:
    concat_parsed(pdf_csv, hwp_csv, output_path) : 두 CSV를 병합해 단일 DataFrame 반환
    run_full_pipeline(folder_path, output_dir)    : PDF·HWP 파싱부터 병합까지 일괄 실행

출력 컬럼:
    ["파일명", "파일형식", "raw_text", "clean_text"]

파이프라인:
    pdf_parser.parse_pdf_folder  ─┐
                                   ├─ concat_parsed → df_parsed.csv
    hwp_parser.parse_hwp_folder_raw ─┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from parsing.pdf_parser import parse_pdf_folder
from parsing.hwp_parser import parse_hwp_folder_raw


# ---------------------------------------------------------------------------
# CSV 기반 병합
# ---------------------------------------------------------------------------

def concat_parsed(
    pdf_csv: str,
    hwp_csv: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    파싱 결과 CSV 두 개를 읽어 단일 DataFrame으로 병합한다.

    PDF CSV 는 rag_text 컬럼을 clean_text 로 사용한다 (make_rag_text 적용본).
    HWP CSV 는 clean_text 컬럼을 그대로 사용한다.

    Args:
        pdf_csv     : df_parsed_pdf_hard.csv 경로
        hwp_csv     : df_parsed_hwp.csv 경로
        output_path : 지정 시 병합 결과를 해당 경로에 저장한다.

    Returns:
        columns = ["파일명", "파일형식", "raw_text", "clean_text"]
    """
    df_pdf = pd.read_csv(pdf_csv, encoding="utf-8")
    df_pdf = df_pdf[["파일명", "파일형식", "raw_text", "rag_text"]].copy()
    df_pdf = df_pdf.rename(columns={"rag_text": "clean_text"})

    df_hwp = pd.read_csv(hwp_csv, encoding="utf-8")
    df_hwp = df_hwp[["파일명", "파일형식", "raw_text", "clean_text"]].copy()

    df_merge = pd.concat([df_pdf, df_hwp], ignore_index=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_merge.to_csv(output_path, index=False, encoding="utf-8")
        print(f"병합 완료: {output_path}  (총 {len(df_merge)}건)")

    return df_merge


# ---------------------------------------------------------------------------
# 풀 파이프라인 (파싱 → 정제 → 병합)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    folder_path: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    PDF·HWP 파싱부터 최종 병합까지 한 번에 실행한다.

    Args:
        folder_path : PDF/HWP 파일들이 있는 디렉토리 경로.
        output_dir  : 중간 CSV 및 최종 df_parsed.csv 저장 디렉토리.

    Returns:
        columns = ["파일명", "파일형식", "raw_text", "clean_text"]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("[1/3] PDF 파싱 중...")
    df_pdf = parse_pdf_folder(folder_path, output_dir=str(out))

    print("=" * 50)
    print("[2/3] HWP 파싱 중...")
    df_hwp = parse_hwp_folder_raw(folder_path, output_dir=str(out))

    print("=" * 50)
    print("[3/3] 병합 중...")
    df_pdf_slim = df_pdf[["파일명", "파일형식", "raw_text", "rag_text"]].rename(
        columns={"rag_text": "clean_text"}
    )
    df_hwp_slim = df_hwp[["파일명", "파일형식", "raw_text", "clean_text"]].copy()

    df_merge = pd.concat([df_pdf_slim, df_hwp_slim], ignore_index=True)

    output_path = out / "df_parsed.csv"
    df_merge.to_csv(output_path, index=False, encoding="utf-8")
    print(f"최종 저장 완료: {output_path}  (총 {len(df_merge)}건)")
    print("=" * 50)

    return df_merge
