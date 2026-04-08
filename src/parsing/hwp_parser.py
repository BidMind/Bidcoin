"""
hwp_parser.py — HWP 파일 파싱 및 정제 모듈

주요 함수:
    parse_hwp_folder_raw(folder_path, output_dir) : 폴더 내 HWP 전체를 파싱 → DataFrame
    parse_single_hwp_raw(file_path)               : HWP 파일 1개 파싱 → dict
    parse_hwp(file_path)                          : HWP raw 텍스트 추출 → str
    clean_hwp_text(text)                          : raw_text → clean_text

파이프라인:
    parse_hwp  →  parse_single_hwp_raw  →  parse_hwp_folder_raw
    raw_text   →  clean_hwp_text
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Optional

import olefile
import pandas as pd


# ---------------------------------------------------------------------------
# (1) HWP 원문 추출
# ---------------------------------------------------------------------------

def parse_hwp(file_path: str) -> str:
    """
    HWP 파일에서 raw_text를 추출한다.

    - 구조 추출·표 추출 없이 rec_type == 67 텍스트 레코드만 사용
    - zlib 압축 해제 포함
    - 파싱 실패 시 빈 문자열 반환
    """
    section_texts = []

    try:
        with olefile.OleFileIO(file_path) as ole:
            if not ole.exists("BodyText"):
                return ""

            section_idx = 0
            while ole.exists(f"BodyText/Section{section_idx}"):
                data = ole.openstream(f"BodyText/Section{section_idx}").read()

                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    pass  # 압축 안 된 경우 그대로 사용

                chars = []
                i = 0

                while i < len(data):
                    if i + 4 > len(data):
                        break

                    header = int.from_bytes(data[i:i + 4], "little")
                    rec_type = header & 0x3FF
                    rec_len = (header >> 20) & 0xFFF
                    i += 4

                    if rec_len == 0xFFF:
                        if i + 4 > len(data):
                            break
                        rec_len = int.from_bytes(data[i:i + 4], "little")
                        i += 4

                    if i + rec_len > len(data):
                        break

                    body = data[i:i + rec_len]
                    i += rec_len

                    if rec_type == 67:
                        for j in range(0, len(body) - 1, 2):
                            ch = int.from_bytes(body[j:j + 2], "little")
                            if ch in (0x0D, 0x0A, 10, 13, 0):
                                chars.append("\n")
                            elif 0x20 <= ch <= 0xD7A3 or ch > 0xE000:
                                try:
                                    chars.append(chr(ch))
                                except ValueError:
                                    pass

                section_text = "".join(chars).strip()
                if section_text:
                    section_texts.append(section_text)

                section_idx += 1

    except Exception:
        return ""

    return "\n".join(section_texts).strip()


# ---------------------------------------------------------------------------
# (2) 단일 파일 파싱
# ---------------------------------------------------------------------------

def parse_single_hwp_raw(file_path: str) -> dict:
    """
    HWP 파일 1개를 raw-only 방식으로 파싱한다.

    Returns:
        {
            "파일명": str, "파일형식": str, "파일경로": str,
            "raw_text": str, "raw_text_len": int,
            "raw_parse_success": bool,
            "status": "success" | "empty" | "error",
            "parse_warning": str | None,
            "processing_time": float,
            "parse_version": str
        }
    """
    start_time = time.time()
    file_name = os.path.basename(file_path)
    file_ext = Path(file_path).suffix.lower().replace(".", "")

    raw_text = ""
    status = "success"
    parse_warning = None

    try:
        raw_text = parse_hwp(file_path)
        if not raw_text:
            status = "empty"
            parse_warning = "raw_text is empty"
    except Exception as e:
        raw_text = ""
        status = "error"
        parse_warning = str(e)

    processing_time = round(time.time() - start_time, 2)
    raw_text_len = len(raw_text) if raw_text else 0

    return {
        "파일명": file_name,
        "파일형식": file_ext,
        "파일경로": str(file_path),
        "raw_text": raw_text,
        "raw_text_len": raw_text_len,
        "raw_parse_success": raw_text_len > 0,
        "status": status,
        "parse_warning": parse_warning,
        "processing_time": processing_time,
        "parse_version": "raw_only_v1",
    }


# ---------------------------------------------------------------------------
# (3) HWP 텍스트 정제
# ---------------------------------------------------------------------------

def clean_hwp_text(text) -> str:
    """
    HWP raw_text에 보수적 공통 정제를 적용한다.

    원칙:
    - 사업명, 예산, 기간, URL, 이메일, 요구사항 코드는 보존
    - 줄 단위 반복 노이즈 + 문장 중간 한자형 노이즈 제거
    - 제목형 띄어쓰기만 제한적으로 복원
    - 숫자/표/항목 구조는 최대한 유지
    """
    if pd.isna(text):
        return text

    text = str(text)
    if not text.strip():
        return text

    # 패턴 정의
    protected_line_pattern = re.compile(
        r"("
        r"(?:https?://|www\.)\S+"
        r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"|(?:TEL|FAX|전화|팩스)\s*[:：]?\s*\+?\d{1,4}[-)\s]?\d{2,4}-\d{3,4}"
        r"|\b[A-Z]{2,5}\s*-\s*\d{2,4}\b"
        r"|(?:[A-Za-z]:)?[\\/][^\s]+"
        r")",
        re.IGNORECASE,
    )

    protected_token_pattern = re.compile(
        r"("
        r"^(?:https?://|www\.)\S+$"
        r"|^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        r"|^(?:TEL|FAX|전화|팩스)$"
        r"|^\+?\d{1,4}[-)\s]?\d{2,4}-\d{3,4}$"
        r"|^[A-Z]{2,5}\s*-\s*\d{2,4}$"
        r"|^(?:[A-Za-z]:)?[\\/].+$"
        r")",
        re.IGNORECASE,
    )

    noise_line_pattern = re.compile(
        r"^\s*(?:"
        r"[氠瑢汤捯桤灧湯湷湰灧漠杳捤獥]+"
        r"|[Āࢀ]+"
        r"|[汫╨]+"
        r"|(?:[^\w가-힣]{1,5})"
        r")\s*$"
    )

    cjk_noise_token_pattern = re.compile(r"^[\u4E00-\u9FFF]{2,6}$")
    cjk_noise_line_pattern = re.compile(r"^\s*(?:[\u4E00-\u9FFF]{2,6}\s*){1,20}$")

    explicit_noise_token_pattern = re.compile(
        r"^(?:氠瑢|汤捯|桤灧|湯湷|湰灧|漠杳|捤獥|Ā|ࢀ|汫╨)$"
    )

    inline_noise_pattern = re.compile(
        r"(?:氠瑢|汤捯|桤灧|湯湷|湰灧|漠杳|捤獥|Ā|ࢀ|汫╨)"
    )

    spaced_ko_title_pattern = re.compile(
        r"(?<![가-힣A-Za-z0-9])(?:[가-힣]\s){1,20}[가-힣](?![가-힣A-Za-z0-9])"
    )

    # 0) 유니코드 정규화
    text = unicodedata.normalize("NFKC", text)

    # 1) 줄바꿈/탭 정리
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # 2) 제어문자 제거
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

    # 3) 자주 보이는 특수 깨짐 문자 제거
    text = re.sub(r"[↸ᬄὩ⇟]", " ", text)

    # 4) 줄 단위 처리
    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        if protected_line_pattern.search(line):
            line = re.sub(r"[ ]{2,}", " ", line).strip()
            cleaned_lines.append(line)
            continue

        if noise_line_pattern.fullmatch(line):
            continue

        if cjk_noise_line_pattern.fullmatch(line):
            continue

        line = inline_noise_pattern.sub(" ", line)

        tokens = line.split()
        kept_tokens = []
        for tok in tokens:
            if protected_token_pattern.search(tok):
                kept_tokens.append(tok)
                continue
            if explicit_noise_token_pattern.fullmatch(tok):
                continue
            if cjk_noise_token_pattern.fullmatch(tok):
                continue
            kept_tokens.append(tok)

        line = " ".join(kept_tokens)
        line = re.sub(r"(?:[‧·•∙]{3,})", " ", line)
        line = re.sub(r"[ ]{2,}", " ", line).strip()

        if not line:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 5) 제목형 띄어쓰기 복원
    def restore_spaced_title(match):
        s = match.group(0)
        compact = s.replace(" ", "")
        if not (2 <= len(compact) <= 25):
            return s
        if re.search(r"(https?|www\.|@|\d{2,4}-\d{2,4}-\d{3,4})", s, re.I):
            return s
        if re.search(r"\b[A-Z]{2,5}\s*-\s*\d{2,4}\b", s):
            return s
        if re.fullmatch(r"(?:[가-힣]\s){1,20}[가-힣]", s.strip()):
            return compact
        return s

    text = spaced_ko_title_pattern.sub(restore_spaced_title, text)

    # 6) 자주 나오는 표제어 보정
    replacements = {
        "제안요청서": "제안요청서",
        "목차": "목차",
        "사업명": "사업명",
        "과업명": "과업명",
        "사업기간": "사업기간",
        "사업예산": "사업예산",
        "제안요청내용": "제안요청 내용",
        "제안요청사항": "제안요청 사항",
        "사업개요": "사업개요",
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{k}\b", v, text)

    # 7) 목차 줄 끝 페이지 번호 제거
    final_lines = []
    for line in text.split("\n"):
        if protected_line_pattern.search(line):
            final_lines.append(line)
            continue
        line = re.sub(r"(\s*[-·•‧∙]\s*\d{1,3})$", "", line).rstrip()
        final_lines.append(line)
    text = "\n".join(final_lines)

    # 8) 빈 줄 과다 축소
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# (4) 폴더 단위 파싱 엔트리포인트
# ---------------------------------------------------------------------------

def parse_hwp_folder_raw(
    folder_path: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    폴더 내 모든 .hwp 파일을 파싱하고 정제해 DataFrame을 반환한다.

    Args:
        folder_path : HWP 파일들이 있는 디렉토리 경로.
        output_dir  : 지정 시 df_parsed_hwp.csv 를 해당 디렉토리에 저장한다.

    Returns:
        columns = ["파일명", "파일형식", "파일경로", "raw_text", "clean_text",
                   "raw_text_len", "clean_text_len", "raw_parse_success",
                   "status", "parse_warning", "processing_time", "parse_version"]
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {folder_path}")

    hwp_files = sorted([f for f in folder.rglob("*.hwp") if f.is_file()])
    print(f"대상 HWP 파일 수: {len(hwp_files)}")

    results = []
    for i, file_path in enumerate(hwp_files, 1):
        print(f"[{i}/{len(hwp_files)}] {file_path.name}")
        row = parse_single_hwp_raw(str(file_path))
        row["clean_text"] = clean_hwp_text(row["raw_text"])
        row["clean_text_len"] = len(row["clean_text"]) if row["clean_text"] else 0
        results.append(row)
        print(
            f"  status={row['status']}  "
            f"raw_len={row['raw_text_len']}  "
            f"clean_len={row['clean_text_len']}  "
            f"time={row['processing_time']}s"
        )

    df = pd.DataFrame(results)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "df_parsed_hwp.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"저장 완료: {csv_path}  ({len(df)}건)")

    return df
