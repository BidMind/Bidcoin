from __future__ import annotations

import os
import re
import zlib
import json
import uuid
import time
import shutil
import signal
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import unicodedata

import olefile
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

from config import DATABASE_DIR, OUTPUT_DIR


# ============================================================
# (0) 공통 유틸
# ============================================================

def make_doc_id(file_path: str) -> str:
    """문서 고유 ID 생성"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(file_path)))


def safe_json_dumps(obj: Any) -> str:
    """JSON 문자열 직렬화"""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def normalize_whitespace(text: str) -> str:
    """약한 정제: 공백 / 개행 / 제어문자 정리"""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # 줄바꿈 통일
    text = text.replace("\t", " ")                           # 탭 → 공백
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)  # 제어문자 제거
    text = "\n".join(line.strip() for line in text.split("\n"))  # 줄 끝 공백 제거
    text = re.sub(r"[ ]{2,}", " ", text)                    # 다중 공백 축소
    text = re.sub(r"\n{3,}", "\n\n", text)                  # 과도한 빈 줄 축소
    return text.strip()


def count_korean_ratio(text: str) -> float:
    """한글 비율 계산"""
    if not text:
        return 0.0
    total = len(text)
    korean = len(re.findall(r"[가-힣]", text))
    return korean / total if total > 0 else 0.0


def count_special_ratio(text: str) -> float:
    """특수문자 비율 계산"""
    if not text:
        return 0.0
    total = len(text)
    special = len(re.findall(r"[^0-9A-Za-z가-힣\s\.\,\(\)\[\]\-_/:%]", text))
    return special / total if total > 0 else 0.0


def is_probable_heading(line: str) -> bool:
    """
    제목/소제목으로 보이는 줄인지 판단
    (structured parser fallback용)
    """
    if not line:
        return False
    line = line.strip()
    if len(line) > 120:
        return False
    heading_patterns = [
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.\)]?\s*.+",
        r"^\d+[.\)]\s*.+",
        r"^[가-힣][.\)]\s*.+",
        r"^(제\s*\d+\s*장)\s*.*",
        r"^(제\s*\d+\s*절)\s*.*",
        r"^(사업개요|사업 내용|사업내용|과업 내용|과업내용|제안요청 내용|제안 요청 내용"
        r"|입찰 참가 자격|입찰참가자격|평가 항목|평가항목)\b.*",
    ]
    for p in heading_patterns:
        if re.match(p, line):
            return True
    return False


def split_paragraphs(text: str) -> List[str]:
    """문단 단위 분리"""
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def rows_to_markdown(rows: List[List[str]]) -> str:
    """2차원 배열 → markdown table"""
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in rows]
    header = normalized[0]
    body = normalized[1:]
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in body:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


# ============================================================
# (1) raw_text 추출 — olefile 기반
# ============================================================

def parse_hwp2(file_path: str) -> str:
    """
    HWP에서 raw_text 추출
    - rec_type == 67 텍스트 레코드 중심
    """
    text = ""
    try:
        with olefile.OleFileIO(file_path) as ole:
            if not ole.exists("BodyText"):
                return ""

            section_idx = 0
            while ole.exists(f"BodyText/Section{section_idx}"):
                data = ole.openstream(f"BodyText/Section{section_idx}").read()
                try:
                    data = zlib.decompress(data, -15)
                except Exception:
                    pass

                result = ""
                i = 0
                while i < len(data):
                    if i + 4 > len(data):
                        break
                    header = int.from_bytes(data[i:i+4], "little")
                    rec_type = header & 0x3FF
                    rec_len = (header >> 20) & 0xFFF

                    if rec_len == 0xFFF:
                        if i + 8 > len(data):
                            break
                        rec_len = int.from_bytes(data[i+4:i+8], "little")
                        i += 4

                    i += 4
                    body = data[i:i+rec_len]
                    i += rec_len

                    if rec_type == 67:
                        for j in range(0, len(body)-1, 2):
                            ch = int.from_bytes(body[j:j+2], "little")
                            if ch in (0x0D, 0x0A, 10, 13, 0):
                                result += "\n"
                            elif 0x20 <= ch <= 0xD7A3 or ch > 0xE000:
                                try:
                                    result += chr(ch)
                                except Exception:
                                    pass

                text += result + "\n"
                section_idx += 1

    except Exception as e:
        print(f"[HWP 오류] {file_path}: {e}")

    return text.strip()


# ============================================================
# (2) 구조 추출용 변환기 — pyhwp(hwp5html)
# ============================================================

def detect_hwp_converter() -> Optional[str]:
    """
    구조 추출용 외부 도구 탐지
    - hwp5html: pyhwp 패키지 설치 시 제공
    """
    return "hwp5html" if shutil.which("hwp5html") else None


# timeout 파라미터 — parse_hwp_folder2의 timeout_sec과 연동
def convert_hwp_to_html(file_path: str, out_dir: str, timeout: int = 1800) -> Optional[str]:
    """
    HWP → HTML 변환 (hwp5html 사용)
    - hwp5html이 없으면 None 반환 → parse_hwp_structured에서 fallback 처리

    Parameters
    ----------
    timeout : subprocess 최대 실행 시간(초). parse_hwp_folder2의 timeout_sec과 연동.
    """
    converter = detect_hwp_converter()
    if converter != "hwp5html":
        print(f"[hwp5html 미설치] converter={converter}")
        return None

    try:
        subprocess.run(
            ["hwp5html", file_path],
            cwd=out_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # print(f"[hwp5html 성공] stdout={result.stdout[:200]}")
    except subprocess.CalledProcessError as e:
        # print(f"[hwp5html 실패] returncode={e.returncode}")
        # print(f"[hwp5html stderr] {e.stderr[:500]}")
        return None
    except subprocess.TimeoutExpired:
        # print(f"[hwp5html 타임아웃] {file_path}")
        return None
    except Exception as e:
        # print(f"[hwp5html 기타 오류] {type(e).__name__}: {e}")
        return None    

    html_files = list(Path(out_dir).rglob("*.xhtml")) + list(Path(out_dir).rglob("*.html"))
    # print(f"[hwp5html html파일 목록] {html_files}")
    if not html_files:
        return None

    return str(html_files[0])


# ============================================================
# (3) HTML 기반 구조 파싱
# ============================================================

def inspect_html_tags(html_text: str, top_n: int = 30) -> Dict[str, Any]:
    """
    HTML 내 태그 분포 확인
    """
    soup = BeautifulSoup(html_text, "html.parser")
    tag_counter = Counter(el.name for el in soup.find_all())

    return {
        "tag_counter": dict(tag_counter),
        "top_tags": tag_counter.most_common(top_n),
    }


def choose_html_tags_from_inspection(tag_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    태그 분포를 보고 구조 추출에 사용할 태그 그룹 선택
    """
    available_tags = set(tag_info.get("tag_counter", {}).keys())

    heading_tags = [t for t in ["h1", "h2", "h3", "h4", "h5", "h6"] if t in available_tags]
    text_tags    = [t for t in ["p", "li"] if t in available_tags]
    table_tags   = [t for t in ["table"] if t in available_tags]

    # h 태그가 아예 없고 div만 많은 경우를 대비한 fallback
    if not heading_tags and "div" in available_tags:
        text_tags = list(dict.fromkeys(text_tags + ["div"]))

    if not text_tags and "span" in available_tags:
        text_tags = list(dict.fromkeys(text_tags + ["span"]))

    return {
        "heading_tags": heading_tags,
        "text_tags": text_tags,
        "table_tags": table_tags,
        "all_target_tags": heading_tags + text_tags + table_tags,
    }


def is_layout_table(rows: List[List[str]]) -> bool:
    """
    레이아웃용 표 판별 → True면 필터링 대상
    """
    if not rows:
        return True
    max_cols = max(len(r) for r in rows)
    total_cells = sum(len(r) for r in rows)
    total_text = sum(len(cell) for r in rows for cell in r)
    all_empty = all(cell.strip() == "" for r in rows for cell in r)

    if max_cols <= 1:
        return True
    if max_cols > 20:  # 열이 비정상적으로 많으면 레이아웃용 (HWP 병합 셀 처리 부산물)
        return True
    if total_cells <= 4 and total_text < 50:
        return True
    if all_empty:
        return True
    
    # 빈 셀 비율 70% 초과면 레이아웃용
    empty_cells = sum(1 for r in rows for cell in r if cell.strip() == "")
    empty_ratio = empty_cells / total_cells if total_cells > 0 else 1.0
    if empty_ratio > 0.7:
        return True

    return False


def parse_html_to_structured(
    html_text: str,
    heading_tags: Optional[List[str]] = None,
    text_tags: Optional[List[str]] = None,
    table_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    HTML → structured_text / sections / tables 추출
    - hwp5html은 h태그 없이 <p><span> 구조로 제목 표현
      → p 태그 텍스트에 is_probable_heading() 적용해서 제목 감지
    - 레이아웃용 표(1열, 셀 수 적음 등)는 is_layout_table()로 필터링
    """
    soup = BeautifulSoup(html_text, "html.parser")

    heading_tags = heading_tags or ["h1", "h2", "h3", "h4"]
    text_tags = text_tags or ["p", "li"]
    table_tags = table_tags or ["table"]

    target_tags = heading_tags + text_tags + table_tags

    structured_lines: List[str] = []
    sections: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    current_title = None
    current_content: List[str] = []
    table_seq = 1

    def flush_section():
        nonlocal current_title, current_content
        if current_title or current_content:
            content = "\n".join(
                [c for c in current_content if str(c).strip()]
            ).strip()
            sections.append({
                "section_title": current_title,
                "content": content,
            })
        current_title = None
        current_content = []

    for el in soup.find_all(target_tags):
        tag = el.name.lower()

        if tag in heading_tags:
            flush_section()
            current_title = el.get_text(" ", strip=True)
            if current_title:
                structured_lines.append(f"## {current_title}")

        elif tag in text_tags:
            # 표 내부 텍스트 중복 수집 방지
            if el.find_parent("table") is not None:
                continue
            txt = normalize_whitespace(el.get_text(" ", strip=True))
            if not txt:
                continue

            # hwp5html은 h태그 없이 <p>로 제목 표현
            # → is_probable_heading()으로 제목 패턴 감지
            if tag == "p" and is_probable_heading(txt):
                flush_section()
                current_title = txt
                structured_lines.append(f"## {txt}")
            else:
                current_content.append(txt)
                structured_lines.append(txt)

        elif tag in table_tags:
            rows = []
            for tr in el.find_all("tr"):
                row = [
                    normalize_whitespace(cell.get_text(" ", strip=True))
                    for cell in tr.find_all(["th", "td"])
                ]
                if row:
                    rows.append(row)

            # 레이아웃용 표 필터링
            if not rows or is_layout_table(rows):
                continue

            markdown = rows_to_markdown(rows)
            tables.append({
                "table_id": table_seq,
                "section_title": current_title,
                "headers": rows[0] if rows else [],
                "rows": rows[1:] if len(rows) > 1 else [],
                "raw_rows": rows,
                "markdown": markdown,
            })
            current_content.append(markdown)
            structured_lines.append(f"### 표 {table_seq}")
            structured_lines.append(markdown)
            table_seq += 1

    flush_section()

    structured_text = normalize_whitespace("\n".join(structured_lines))
    return {
        "structured_text": structured_text,
        "sections": sections,
        "tables": tables,
    }


# ============================================================
# (4) raw_text fallback 기반 구조 추출
# ============================================================

def build_sections_from_raw_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    HTML 변환 실패 시 raw_text에서 제목/본문 추정
    - is_probable_heading()으로 제목 패턴 감지
    """
    raw_text = normalize_whitespace(raw_text)
    if not raw_text:
        return []

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    sections: List[Dict[str, Any]] = []
    current_title = None
    current_content: List[str] = []

    def flush():
        nonlocal current_title, current_content
        if current_title or current_content:
            sections.append({
                "section_title": current_title,
                "content": "\n".join(current_content).strip(),
            })
        current_title = None
        current_content = []

    for line in lines:
        if is_probable_heading(line):
            flush()
            current_title = line
        else:
            current_content.append(line)

    flush()

    # 제목이 하나도 안 잡히면 전체를 한 섹션으로
    if not sections and raw_text:
        sections = [{"section_title": None, "content": raw_text}]

    return sections


def build_structured_text_from_sections(
    sections: List[Dict[str, Any]],
) -> str:
    """sections기반 structured_text 재구성"""
    lines: List[str] = []
    for sec in sections:
        title = sec.get("section_title")
        content = sec.get("content", "")
        if title:
            lines.append(f"## {title}")
        if content:
            lines.append(content)
    return normalize_whitespace("\n".join(lines))


def extract_simple_tables_from_raw_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    raw_text만 있을 때 단순 표 후보 추출 (fallback용)
    - |, 탭, 연속 공백 패턴을 표로 간주
    """
    if not raw_text:
        return []

    tables = []
    table_id = 1

    for para in split_paragraphs(raw_text):
        lines = [x.strip() for x in para.split("\n") if x.strip()]
        if len(lines) < 2:
            continue

        tabular_lines = [
            line for line in lines
            if "|" in line or "\t" in line or re.search(r"\s{2,}", line)
        ]

        if len(tabular_lines) < 2:
            continue

        rows = []
        for line in tabular_lines:
            if "|" in line:
                row = [c.strip() for c in line.split("|") if c.strip()]
            elif "\t" in line:
                row = [c.strip() for c in line.split("\t") if c.strip()]
            else:
                row = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
            if row:
                rows.append(row)

        if len(rows) >= 2:
            tables.append({
                "table_id": table_id,
                "section_title": None,
                "headers": rows[0],
                "rows": rows[1:],
                "raw_rows": rows,
                "markdown": rows_to_markdown(rows),
            })
            table_id += 1

    return tables


# ============================================================
# (5) 구조 추출 메인 — HTML 우선, fallback 자동 전환
# ============================================================

# timeout 파라미터 추가 — convert_hwp_to_html로 전달
def parse_hwp_structured(
    file_path: str,
    raw_text: Optional[str] = None,
    timeout: int = 1800,
    inspect_tags: bool = True,
) -> Dict[str, Any]:
    """
    HWP 구조 추출 메인 함수

    우선순위
    --------
    1) hwp5html → HTML 변환 → parse_html_to_structured()
    2) 실패하거나 구조가 빈약하면 raw_text fallback

    Parameters
    ----------
    timeout : hwp5html subprocess 최대 실행 시간(초)

    Returns
    -------
    {
        structured_text, sections, tables,
        html_parse_success,   # hwp5html 변환 성공 여부
        struct_parse_success, # structured_text 존재 여부 (텍스트 기준)
        table_extract_success,
        parse_method, parse_warning
    }
    """
    if raw_text is None:
        raw_text = parse_hwp2(file_path)

    result: Dict[str, Any] = {
        "structured_text": "",
        "sections": [],
        "tables": [],
        "html_parse_success": False,   # hwp5html 변환 성공 여부
        "struct_parse_success": False, # structured_text 존재 여부
        "table_extract_success": False,
        "parse_method": None,
        "parse_warning": "",
        "html_top_tags": None,
        "html_selected_tags": None,
    }

    # 1차: HTML 변환 기반 구조 추출
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = convert_hwp_to_html(file_path, tmpdir, timeout=timeout)

            if html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_text = f.read()

                if inspect_tags:
                    tag_info = inspect_html_tags(html_text)
                    selected = choose_html_tags_from_inspection(tag_info)
                    result["html_top_tags"] = tag_info["top_tags"]
                    result["html_selected_tags"] = selected

                    parsed = parse_html_to_structured(
                        html_text,
                        heading_tags=selected["heading_tags"],
                        text_tags=selected["text_tags"],
                        table_tags=selected["table_tags"],
                    )
                else:
                    parsed = parse_html_to_structured(html_text)

                has_meaningful_structure = bool(
                    parsed["structured_text"] or parsed["sections"] or parsed["tables"]
                )

                if has_meaningful_structure:
                    result.update({
                        "structured_text": parsed["structured_text"],
                        "sections": parsed["sections"],
                        "tables": parsed["tables"],
                        "html_parse_success": True,
                        "struct_parse_success": bool(parsed["structured_text"]),
                        "table_extract_success": len(parsed["tables"]) > 0,
                        "parse_method": "hwp5html",
                    })
                    return result
                else:
                    result["parse_warning"] = "html_generated_but_structure_empty"

    except Exception as e:
        result["parse_warning"] = f"html_struct_fail: {e}"

    # 2차 : raw_text기반 추출
    sections = build_sections_from_raw_text(raw_text)
    tables = extract_simple_tables_from_raw_text(raw_text)
    structured_text = build_structured_text_from_sections(sections)

    result.update({
        "structured_text": structured_text,
        "sections": sections,
        "tables": tables,
        "html_parse_success": False,
        "struct_parse_success": bool(structured_text),
        "table_extract_success": len(tables) > 0,
        "parse_method": "raw_fallback",
    })

    if not structured_text:
        result["parse_warning"] = (
            result["parse_warning"] + " | structured_text_empty"
        ).lstrip(" | ")

    return result



# ============================================================
# (6) 파싱 품질 평가
# ============================================================

def evaluate_parse_quality(
    raw_text: str,
    structured_text: str,
    sections: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    파싱 품질 지표 및 경고 플래그 생성

    경고 종류
    ---------
    raw_text_too_short     : raw_text 300자 미만
    low_korean_ratio       : 한글 비율 5% 미만 (텍스트가 깨진 신호)
    high_special_ratio     : 특수문자 비율 30% 초과
    no_sections_on_long_doc: 1000자 이상인데 섹션 0개
    """
    raw_text = raw_text or ""
    structured_text = structured_text or ""
    sections = sections or []
    tables = tables or []

    raw_text_len = len(raw_text)
    structured_text_len = len(structured_text)
    section_count = len(sections)
    table_count = len(tables)
    korean_ratio = count_korean_ratio(raw_text)
    special_ratio = count_special_ratio(raw_text)

    warnings = []
    if raw_text_len < 300:
        warnings.append("raw_text_too_short")
    if korean_ratio < 0.05 and raw_text_len > 100:
        warnings.append("low_korean_ratio")
    if special_ratio > 0.30 and raw_text_len > 100:
        warnings.append("high_special_ratio")
    if raw_text_len > 1000 and section_count == 0:
        warnings.append("no_sections_on_long_doc")

    return {
        "raw_text_len": raw_text_len,
        "structured_text_len": structured_text_len,
        "section_count": section_count,
        "table_count": table_count,
        "korean_ratio": round(korean_ratio, 4),
        "special_ratio": round(special_ratio, 4),
        "raw_parse_success": raw_text_len > 0,
        "parse_warning": "; ".join(warnings),
    }


# ============================================================
# (6-2) HWP 텍스트 정제
# ============================================================

def clean_hwp_text(text: str) -> str:
    """
    raw_text 기준 보수적 공통 정제

    적용 대상
    ---------
    - parse_method == "raw_fallback" : raw_text가 clean_text가 되므로 반드시 정제
    - parse_method == "hwp5html"     : BeautifulSoup이 이미 처리하므로 적용 불필요

    원칙
    - 사업명, 예산, 기간, URL, 이메일, 요구사항 코드는 보존
    - 제어문자, 줄바꿈, 연속 기호 등 범용 노이즈만 정제
    - 제목형 띄어쓰기만 제한적으로 복원
    - 숫자/표/항목 구조는 최대한 유지
    """
    # None 입력 시 빈 문자열 반환 (타입 불일치 방지)
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(text)
    if not text.strip():
        return ""

    # 보호 패턴 (URL, 이메일, 연락처, 요구사항코드, 경로)
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
    symbol_noise_line_pattern = re.compile(r"^\s*[^\w가-힣A-Za-z0-9]{1,10}\s*$")
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

        # 보호 줄은 그대로
        if protected_line_pattern.search(line):
            line = re.sub(r"[ ]{2,}", " ", line).strip()
            cleaned_lines.append(line)
            continue

        # 줄 전체가 노이즈면 제거
        if symbol_noise_line_pattern.fullmatch(line):
            continue

        # 토큰 단위 처리
        tokens = line.split()
        kept_tokens = []
        for tok in tokens:
            if protected_token_pattern.search(tok):
                kept_tokens.append(tok)
                continue
            kept_tokens.append(tok)
        line = " ".join(kept_tokens)

        # 연속 기호 노이즈 축소
        line = re.sub(r"(?:[‧·•∙]{3,})", " ", line)

        # 공백 정리
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
        "제안요청서": "제안요청서", "목차": "목차", "사업명": "사업명",
        "과업명": "과업명", "사업기간": "사업기간", "사업예산": "사업예산",
        "제안요청내용": "제안요청 내용", "제안요청사항": "제안요청 사항",
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


# ============================================================
# (7) 단일 파일 파싱 — 1행 dict 생성
# ============================================================

# timeout 파라미터 — parse_hwp_structured로 전달
def parse_single_hwp_document(file_path: str, timeout: int = 1800) -> Dict[str, Any]:
    """
    HWP 파일 1개 → 문서 테이블 1행 dict
    메타데이터 merge는 외부(청킹 단계)에서 담당.

    Parameters
    ----------
    timeout : hwp5html subprocess 최대 실행 시간(초)
    """
    file_path = str(file_path)
    file_name = os.path.basename(file_path)
    doc_id = make_doc_id(file_path)

    # raw_text 원본 + 약한 정제본 둘 다 유지
    raw_text_original = parse_hwp2(file_path)
    raw_text = normalize_whitespace(raw_text_original)

    # 구조 추출 (HTML 우선 → fallback)
    structured_result = parse_hwp_structured(file_path, raw_text=raw_text_original, timeout=timeout)  # timeout 전달
    structured_text = normalize_whitespace(structured_result.get("structured_text", ""))
    sections = structured_result.get("sections", [])
    tables = structured_result.get("tables", [])

    # clean_text: parse_method와 structured_text 존재 여부로 정제 방식 분기
    # - hwp5html 성공 + structured_text 있음 : BeautifulSoup이 이미 노이즈 제거 → normalize_whitespace만
    # - 그 외 (raw_fallback 또는 hwp5html이지만 structured_text 비어있음)
    #   : raw_text 기반이므로 clean_hwp_text 정제 적용
    # hwp5html이어도 structured_text가 비어있으면 clean_hwp_text 적용
    base_text = structured_text if structured_text else raw_text
    if structured_result.get("parse_method") == "hwp5html" and structured_text:
        clean_text = normalize_whitespace(base_text)
    else:
        clean_text = clean_hwp_text(base_text)

    # 품질 평가
    quality = evaluate_parse_quality(raw_text, structured_text, sections, tables)

    row: Dict[str, Any] = {
        "doc_id": doc_id,
        "파일명": file_name,
        "파일경로": file_path,
        "파일형식": "hwp",
        "raw_text": raw_text,
        "clean_text": clean_text,
        "structured_text": structured_text,
        "sections": sections,
        "tables": tables,
        "sections_json": safe_json_dumps(sections),
        "tables_json": safe_json_dumps(tables),
        "parse_method": structured_result.get("parse_method"),
        "html_top_tags": structured_result.get("html_top_tags"),
        "html_selected_tags": structured_result.get("html_selected_tags"),
        "raw_parse_success": quality["raw_parse_success"],
        "html_parse_success": structured_result.get("html_parse_success"),   # hwp5html 변환 성공 여부
        "struct_parse_success": structured_result.get("struct_parse_success"),  # structured_text 존재 여부
        "table_extract_success": structured_result.get("table_extract_success"),
        "raw_text_len": quality["raw_text_len"],
        "structured_text_len": quality["structured_text_len"],
        "section_count": quality["section_count"],
        "table_count": quality["table_count"],
        "korean_ratio": quality["korean_ratio"],
        "special_ratio": quality["special_ratio"],
        "parse_warning": " | ".join(
            x for x in [structured_result.get("parse_warning", ""), quality["parse_warning"]] if x
        ),
    }

    return row


# ============================================================
# (8) 타임아웃 설정
# ============================================================

class ParseTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ParseTimeoutError("parse timeout")


def _make_error_row(
    file_name: str,
    file_path: str,
    status: str,
    warning: str,
    elapsed: float,
) -> Dict[str, Any]:
    """타임아웃 / 오류 발생 시 빈 행 생성"""
    return {
        "doc_id": make_doc_id(file_path),
        "파일명": file_name,
        "파일경로": file_path,
        "파일형식": "hwp",
        "raw_text": None,
        "clean_text": None,
        "structured_text": None,
        "sections": None,
        "tables": None,
        "sections_json": None,
        "tables_json": None,
        "parse_method": status,
        "raw_parse_success": False,
        "html_parse_success": False,
        "struct_parse_success": False,
        "table_extract_success": False,
        "raw_text_len": 0,
        "structured_text_len": 0,
        "section_count": 0,
        "table_count": 0,
        "korean_ratio": 0.0,
        "special_ratio": 0.0,
        "parse_warning": warning,
        "processing_time": round(elapsed, 2),
        "status": status,
    }


# ============================================================
# (9) 폴더 전체 파싱 (타임아웃 포함) — 메인 실행 함수
# ============================================================

def parse_hwp_folder2(
    folder_path: str,
    timeout_sec: int = 1800,
    save_error_row: bool = True,
) -> pd.DataFrame:
    """
    HWP 폴더 전체 순차 파싱 (타임아웃 포함)
    메타데이터 merge는 외부(청킹 단계)에서 담당.

    Parameters
    ----------
    folder_path   : HWP 파일이 있는 폴더 경로
    timeout_sec   : 파일 1개당 최대 허용 시간(초). signal.alarm 및 subprocess 양쪽에 적용.
    save_error_row: 타임아웃/오류 파일도 결과 df에 남길지 여부

    Returns
    -------
    pd.DataFrame : 파일 1개 = 1행
    """
    folder = Path(folder_path)
    hwp_files = sorted(folder.rglob("*.hwp"))
    total_files = len(hwp_files)
    print(f"대상 HWP 파일 수: {total_files}")

    rows = []
    total_start = time.time()

    for idx, fp in enumerate(hwp_files, 1):
        file_name = fp.name
        file_path = str(fp)
        file_start = time.time()

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)

        try:
            row = parse_single_hwp_document(file_path, timeout=timeout_sec)  
            elapsed = time.time() - file_start
            row["processing_time"] = round(elapsed, 2)
            row["status"] = "success"
            rows.append(row)
            print(
                f"[{idx}/{total_files}] 완료: {file_name} ({elapsed:.2f}s)\n"
                f"    - parse_method         : {row.get('parse_method')}\n"
                f"    - html_parse_success   : {row.get('html_parse_success')}\n"
                f"    - struct_parse_success : {row.get('struct_parse_success')}\n"
                f"    - table_extract_success: {row.get('table_extract_success')}\n"
                f"    - html_top_tags        : {row.get('html_top_tags')}\n"
                f"    - html_selected_tags   : {row.get('html_selected_tags')}\n"
                f"    - raw_text_len         : {row.get('raw_text_len')}\n"
                f"    - structured_text_len  : {row.get('structured_text_len')}\n"
                f"    - section_count        : {row.get('section_count')}\n"
                f"    - table_count          : {row.get('table_count')}\n"
                f"    - parse_warning        : {row.get('parse_warning') or '-'}"
            )

        except ParseTimeoutError:
            elapsed = time.time() - file_start
            print(f"[{idx}/{total_files}] 타임아웃: {file_name} ({elapsed:.2f}s)")
            if save_error_row:
                rows.append(_make_error_row(
                    file_name, file_path, "timeout",
                    f"{timeout_sec}s timeout", elapsed,
                ))

        except Exception as e:
            elapsed = time.time() - file_start
            print(f"[{idx}/{total_files}] 오류: {file_name} | {e}")
            if save_error_row:
                rows.append(_make_error_row(
                    file_name, file_path, "error",
                    str(e), elapsed,
                ))

        finally:
            signal.alarm(0)  # 타임아웃 해제

    total_elapsed = time.time() - total_start
    print(f"\n전체 완료: {total_elapsed:.2f}s")

    df = pd.DataFrame(rows)

    if not df.empty and "status" in df.columns:
        print("\nstatus 요약")
        print(df["status"].value_counts(dropna=False))

    return df


# ============================================================
# 단독 실행
# ============================================================

if __name__ == "__main__":
    # 1. 경로 설정
    folder_path = DATABASE_DIR / "files"  # HWP 파일 폴더
    output_path = OUTPUT_DIR              # 저장 경로

    csv_path = output_path / "df_parsed_hwp_v2.csv"
    pkl_path = output_path / "df_parsed_hwp_v2.pkl"

    # 2. 전체 파싱 실행
    df = parse_hwp_folder2(
        folder_path=folder_path,
        timeout_sec=1800,
        save_error_row=True,
    )

    # 3. 저장
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_pickle(pkl_path)
    print(f"\n저장 완료: OUTPUT_DIR/{csv_path.name}")
    print(f"저장 완료: OUTPUT_DIR/{pkl_path.name}")
    print(df[["파일명", "parse_method", "status", "raw_text_len", "structured_text_len"]].head(10))
