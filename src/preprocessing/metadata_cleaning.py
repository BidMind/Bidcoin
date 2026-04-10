"""
preprocessing/metadata_cleaning.py

메타데이터 정제 모듈
- (1) 파일명 / 파일형식 수정  (hwp→pdf 변환 과정에서 바뀐 파일명 매핑)
- (2) 결측처리               (입찰 참여 시작일, 마감일, 사업 금액)
- (3) 텍스트 재파싱           (PDF: fitz / HWP: olefile)

사용 예시
---------
from src.preprocessing.metadata_cleaning import process_metadata

df = pd.read_csv("data_list.csv", encoding="utf-8")
df = process_metadata(df, files_dir="/home/shared/files")
"""

from __future__ import annotations

import os
import re
import zlib
import unicodedata
from typing import Optional

import pandas as pd
import fitz          # pymupdf
import olefile

from pathlib import Path
from dotenv import load_dotenv
from config import DATABASE_DIR, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parent  
load_dotenv(ROOT_DIR / ".env")

# ============================================================
# (1) 파일명 / 파일형식 수정
# ============================================================

# hwp→pdf 변환 과정에서 파일명이 바뀐 케이스 매핑
_RENAME_MAP: dict[str, str] = {
    "한국농어촌공사_아세안+3 식량안보정보시스템(AFSIS) 3단계 협력(캄보디아.hwp":
        "한국농어촌공사_아세안+3+식량안보정보시스템(AFSIS)+3단계+협력(캄보디아.hwp.pdf",
    "대전대학교_대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전.hwp":
        "대전대학교_대전대학교+2024학년도+다층적+융합+학습경험+플랫폼(MILE)+전.hwp.pdf",
}

# 위 파일들은 파일형식도 pdf로 교정
_RENAME_TARGETS = list(_RENAME_MAP.values())


def _fix_filename_and_format(df: pd.DataFrame) -> pd.DataFrame:
    """파일명 rename + 파일형식 pdf 교정."""
    df["파일명"] = df["파일명"].replace(_RENAME_MAP)
    df.loc[df["파일명"].isin(_RENAME_TARGETS), "파일형식"] = "pdf"
    return df


# ============================================================
# (2) 결측처리
# ============================================================

# 사업 금액 직접 입력값
_AMOUNT_OVERRIDES: dict[str, int] = {
    "서울시립대학교_[사전공개] 학업성취도 다차원 종단분석 통합시스템 1차.pdf": 242900000,
    "경희대학교_[입찰공고] 산학협력단 정보시스템 운영 용역업체 선정.hwp":     200000000,
    "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp":                  470000000,
    "한국철도공사 (용역)_모바일오피스 시스템 고도화 용역(총체 및 1차).hwp":    843000000,
    "사단법인 보험개발원_실손보험 청구 전산화 시스템 구축 사업.hwp":                    0,
}

# 마감일 직접 입력값
_DEADLINE_OVERRIDES: dict[str, str] = {
    "경희대학교_[입찰공고] 산학협력단 정보시스템 운영 용역업체 선정.hwp": "2024-05-14 11:00:00",
}


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """결측처리: 시작일 → 공개일자 대체, 마감일/사업금액 → 직접 입력."""

    # (2-1) 입찰 참여 시작일: 결측이면 공개 일자로 대체
    mask_start = df["입찰 참여 시작일"].isna() & df["공개 일자"].notna()
    df.loc[mask_start, "입찰 참여 시작일"] = df.loc[mask_start, "공개 일자"]

    # (2-2) 입찰 참여 마감일: 특정 파일 직접 입력
    for fname, val in _DEADLINE_OVERRIDES.items():
        df.loc[df["파일명"] == fname, "입찰 참여 마감일"] = val

    # (2-3) 사업 금액: 전체 결측 → 57,000,000 기본값 후 개별 override
    df.loc[df["사업 금액"].isna(), "사업 금액"] = 57000000
    for fname, val in _AMOUNT_OVERRIDES.items():
        df.loc[df["파일명"] == fname, "사업 금액"] = val

    return df


# ============================================================
# (3-1) PDF 텍스트 재파싱
# ============================================================

def _extract_pdf_text(file_path: str) -> str:
    """fitz(PyMuPDF)로 PDF 전체 텍스트 추출."""
    with fitz.open(file_path) as doc:
        return "".join(page.get_text() for page in doc)


def _build_spaced_phrase_pattern(phrase: str) -> str:
    """띄어쓰기로 깨진 단어 복구용 정규식 패턴 생성."""
    chars = [re.escape(ch) for ch in phrase if ch.strip()]
    return r"(?<![가-힣A-Za-z0-9])" + r"\s*".join(chars) + r"(?![가-힣A-Za-z0-9])"


def _remove_unwanted_characters(text: str) -> tuple[str, int]:
    """허용 문자 외 문자를 공백으로 변환."""
    pattern = r"[^가-힣a-zA-Z0-9\s\.\(\)\[\]\/\,\%\:\-\·\?\!\@\&\+]"
    return re.subn(pattern, " ", text)


def _remove_page_markers(text: str) -> tuple[str, int]:
    """쪽번호 패턴(-N-) 제거."""
    pattern = r"(?<![\dA-Za-z가-힣])-\s*\d{1,3}\s*-(?![\dA-Za-z가-힣])"
    return re.subn(pattern, " ", text)


def _remove_repeated_markers(
    text: str,
    markers: Optional[list[str]] = None,
) -> tuple[str, dict]:
    """반복 표식(예: [사전공개용]) 제거."""
    if markers is None:
        markers = ["[사전공개용]"]
    counts: dict[str, int] = {}
    for marker in markers:
        text, count = re.subn(re.escape(marker), " ", text)
        counts[marker] = count
    return text, counts


def _normalize_spacing_and_symbols(text: str) -> tuple[str, dict]:
    """점선(……), 구분선(---), 연속 공백 정리."""
    counts: dict[str, int] = {}
    text, c1 = re.subn(r"[\.·]{2,}", " ", text)
    counts["dot_leaders_removed"] = c1
    text, c2 = re.subn(r"[-_=]{3,}", " ", text)
    counts["repeated_symbols_removed"] = c2
    text, c3 = re.subn(r"\s+", " ", text)
    counts["whitespace_normalized"] = c3
    return text.strip(), counts


_COMMON_PHRASES = [
    "제안요청서", "제안서", "사업명", "사업비", "사업기간", "발주기관",
    "입찰", "계약", "과업내용", "과업지시서", "평가항목", "평가기준",
    "제안안내", "참가자격", "사업개요", "추진배경", "추진방안", "추진일정",
    "요구사항", "기능요구사항", "인터페이스요구사항", "유지보수", "통합시스템",
]


def _fix_spaced_phrases(
    text: str,
    phrases: Optional[list[str]] = None,
) -> tuple[str, dict]:
    """PDF에서 자주 깨지는 핵심 용어 복구."""
    if phrases is None:
        phrases = _COMMON_PHRASES
    counts: dict[str, int] = {}
    for phrase in phrases:
        pattern = _build_spaced_phrase_pattern(phrase)
        text, count = re.subn(pattern, phrase, text)
        counts[phrase] = count
    return text, counts


_BODY_PATTERNS = [
    r"Ⅰ\.\s*사업개요", r"1\.\s*사업개요",
    r"Ⅰ\.\s*과업개요", r"1\.\s*과업개요",
    r"Ⅰ\.\s*제안개요", r"1\.\s*제안개요",
    r"Ⅰ\.\s*추진배경", r"1\.\s*추진배경",
    r"Ⅰ\.\s*일반사항", r"1\.\s*일반사항",
    r"Ⅰ\.\s*제안안내", r"1\.\s*제안안내",
]


def _remove_front_toc_block(text: str) -> tuple[str, dict]:
    """문서 앞부분의 목차 블록 제거."""
    report = {
        "toc_removed": False,
        "toc_removed_chars": 0,
        "body_start_pattern": None,
    }
    toc_match = re.search(r"목\s*차", text[:5000])
    if not toc_match:
        return text, report

    toc_start = toc_match.start()
    search_area = text[toc_start: min(len(text), toc_start + 12000)]
    candidates: list[tuple[int, str]] = []

    for pat in _BODY_PATTERNS:
        for m in re.finditer(pat, search_area):
            abs_pos = toc_start + m.start()
            if abs_pos > toc_start + 200:
                candidates.append((abs_pos, pat))

    if not candidates:
        return text, report

    real_body_start, used_pattern = sorted(candidates, key=lambda x: x[0])[0]

    if toc_start < real_body_start and toc_start < 3000:
        report["toc_removed"] = True
        report["toc_removed_chars"] = real_body_start - toc_start
        report["body_start_pattern"] = used_pattern
        text = text[:toc_start].rstrip() + " " + text[real_body_start:].lstrip()

    return text, report


_ADDITIONAL_REPLACEMENTS: dict[str, str] = {
    "시 스템": "시스템",
    "인 프라": "인프라",
    "정 량적": "정량적",
    "정 성적": "정성적",
    "기 능":   "기능",
    "요 구사항": "요구사항",
    "유 지보수": "유지보수",
    "데 이터": "데이터",
    "서 비스": "서비스",
    "사 업":   "사업",
    "계 약":   "계약",
}


def _fix_additional_phrases(
    text: str,
    replacements: Optional[dict[str, str]] = None,
) -> tuple[str, dict]:
    """어색한 표현 수동 치환."""
    if replacements is None:
        replacements = _ADDITIONAL_REPLACEMENTS
    counts: dict[str, int] = {}
    for old, new in replacements.items():
        text, count = re.subn(re.escape(old), new, text)
        counts[f"{old} -> {new}"] = count
    return text, counts


def _clean_pdf_text(text: str) -> tuple[str, dict]:
    """PDF 텍스트 전처리 통합 함수."""
    report: dict = {"before_length": len(text)}

    text = unicodedata.normalize("NFC", text)
    text, c = _remove_unwanted_characters(text)
    report["unwanted_characters_removed"] = c
    text, c = _remove_page_markers(text)
    report["page_markers_removed"] = c
    text, d = _remove_repeated_markers(text)
    report["marker_removed_counts"] = d
    text, d = _normalize_spacing_and_symbols(text)
    report["spacing_symbol_counts"] = d
    text, d = _fix_spaced_phrases(text)
    report["spaced_phrase_fixed_counts"] = d
    text, d = _remove_front_toc_block(text)
    report["toc_report"] = d
    text, d = _fix_additional_phrases(text)
    report["additional_phrase_counts"] = d

    report["after_length"] = len(text)
    return text, report


def _reparse_pdf(df: pd.DataFrame, files_dir: str) -> pd.DataFrame:
    """df 중 파일형식==pdf인 행의 텍스트를 재파싱하여 갱신."""
    target_files = (
        df.loc[df["파일형식"].str.lower() == "pdf", "파일명"]
        .dropna()
        .unique()
        .tolist()
    )

    for file_name in target_files:
        file_path = os.path.join(files_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[PDF 건너뜀] 파일 없음: {file_name}")
            continue
        try:
            raw = _extract_pdf_text(file_path)
            cleaned, report = _clean_pdf_text(raw)
            mask = df["파일명"] == file_name
            df.loc[mask, "텍스트"] = cleaned
            df.loc[mask, "텍스트길이"] = len(cleaned)
            print(
                f"[PDF 완료] {file_name} | "
                f"원본={len(raw):,} → 정제={len(cleaned):,} | "
                f"목차제거={report['toc_report']['toc_removed']}"
            )
        except Exception as e:
            print(f"[PDF 오류] {file_name}: {e}")

    return df


# ============================================================
# (3-2) HWP 텍스트 재파싱
# ============================================================

def _is_clean_hangul(char: str) -> bool:
    """cp949 기준으로 깨지지 않은 한글인지 판별."""
    if not ("가" <= char <= "힣"):
        return True
    try:
        code = char.encode("cp949")
        return 0xB0 <= code[0] <= 0xC8
    except Exception:
        return False


_STOPWORD_JOSA = {"이", "가", "을", "를", "에", "와", "과", "도", "의", "는", "은", "로", "만", "한"}


def _get_hwp_text(file_path: str) -> Optional[dict]:
    """olefile로 HWP BodyText 섹션 추출 후 보수적 정제."""
    try:
        with olefile.OleFileIO(file_path) as f:
            dirs = f.listdir()
            bodytext_sections = [d for d in dirs if "BodyText" in d]

            raw_text = ""
            for section in bodytext_sections:
                data = f.openstream(section).read()
                try:
                    decompressed = zlib.decompress(data, -15)
                except Exception:
                    decompressed = data
                raw_text += decompressed.decode("utf-16", errors="ignore")

        raw_text = unicodedata.normalize("NFC", raw_text)
        original_length = len(raw_text)

        # 이미지/도형 메타 텍스트 제거
        text = re.sub(r"원본 그림의 이름:.*?pixel", " ", raw_text, flags=re.DOTALL)
        text = re.sub(r"가로 \d+pixel,\s*세로 \d+pixel", " ", text)

        # 허용 문자 외 제거 (PDF와 동일 기준)
        text = re.sub(r"[^가-힣a-zA-Z0-9\s\.\(\)\[\]\/\,\%\:\-\·\?\!\@\&\+]", " ", text)

        # 과도한 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        # 토큰 단위 약한 정제 (cp949 기준으로 명백히 깨진 토큰, 단독 조사 제거)
        clean_tokens = []
        for token in text.split():
            if not all(_is_clean_hangul(c) for c in token):
                continue
            if re.fullmatch(r"[가-힣]", token) and token in _STOPWORD_JOSA:
                continue
            clean_tokens.append(token)

        cleaned_text = " ".join(clean_tokens)
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "original_length": original_length,
            "cleaned_length": len(cleaned_text),
        }

    except Exception as e:
        print(f"[HWP 추출 실패] {file_path}: {e}")
        return None


def _reparse_hwp(df: pd.DataFrame, files_dir: str) -> pd.DataFrame:
    """df 중 파일형식==hwp인 행의 텍스트를 재파싱하여 갱신."""
    target_files = (
        df.loc[df["파일형식"].astype(str).str.strip().str.lower() == "hwp", "파일명"]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"HWP 재파싱 시작 (총 {len(target_files)}개)")
    print("-" * 60)

    for file_name in target_files:
        file_path = os.path.join(files_dir, file_name)
        idx_list = df[df["파일명"] == file_name].index

        if idx_list.empty:
            print(f"[HWP 건너뜀] df에서 파일명 없음: {file_name}")
            continue
        if not os.path.exists(file_path):
            print(f"[HWP 건너뜀] 파일 없음: {file_name}")
            continue

        result = _get_hwp_text(file_path)
        if result is None:
            continue

        cleaned_text = result["cleaned_text"]
        print(
            f"[HWP] {file_name} | "
            f"원본={result['original_length']:,} → 정제={result['cleaned_length']:,}"
        )

        if cleaned_text and result["cleaned_length"] > 100:
            df.loc[idx_list[0], "텍스트"] = cleaned_text
            df.loc[idx_list[0], "텍스트길이"] = result["cleaned_length"]
        else:
            print(f"  └ 텍스트가 너무 짧아 갱신 생략")

    print("-" * 60)
    print("HWP 재파싱 완료")
    return df


# ============================================================
# 메인 함수
# ============================================================

def process_metadata(
    input_csv_path: Path = DATABASE_DIR / "data_list.csv",
    files_dir: Path = DATABASE_DIR / "files",
    output_csv_path: Path = OUTPUT_DIR / "data_list_metadata.csv",
) -> pd.DataFrame:
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    """
    메타데이터 정제 전체 파이프라인.
    - 내부에서 CSV를 직접 읽음
    - 처리 후 /home/bidcoin/data_list_metadata.csv 로 저장
    """
    df = pd.read_csv(input_csv_path, encoding="utf-8")
    df = df.copy()

    print("=" * 60)
    print(f"원본 CSV 로드: {input_csv_path}")
    print(f"행 개수: {len(df)}")

    # (1) 파일명 / 파일형식 수정
    print("=" * 60)
    print("(1) 파일명 / 파일형식 수정")
    df = _fix_filename_and_format(df)

    # (2) 결측처리
    print("=" * 60)
    print("(2) 결측처리 (시작일 / 마감일 / 사업 금액)")
    df = _fill_missing_values(df)
    print(f"  입찰 참여 시작일 결측 잔여: {df['입찰 참여 시작일'].isna().sum()}")
    print(f"  입찰 참여 마감일 결측 잔여: {df['입찰 참여 마감일'].isna().sum()}")
    print(f"  사업 금액 결측 잔여:        {df['사업 금액'].isna().sum()}")

    # (3) 텍스트 재파싱
    print("=" * 60)
    print("(3-1) PDF 텍스트 재파싱")
    df = _reparse_pdf(df, files_dir)

    print("=" * 60)
    print("(3-2) HWP 텍스트 재파싱")
    df = _reparse_hwp(df, files_dir)

    # 저장
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print("=" * 60)
    print("process_metadata 완료")
    print(f"저장 완료: {output_csv_path}")
    return df

if __name__ == "__main__":
    process_metadata()

# main에서 메타데이터 로드 필요없어짐
#ex.
# df = pd.read_csv("data_list.csv", encoding="utf-8") 
# df = process_metadata(df, files_dir="/home/shared/files")
# df.to_csv("data_list_metadata.csv", index=False, encoding="utf-8")
#수정후.
# from preprocessing.metadata_cleaning import process_metadata
# df = process_metadata()만 하면됨