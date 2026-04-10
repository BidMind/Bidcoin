"""
pdf_parser.py — PDF 파일 파싱 및 정제 모듈

주요 함수:
    parse_pdf_folder(folder_path, output_dir) : 폴더 내 PDF 전체를 파싱 → DataFrame
    clean_pdf_text(text)                      : raw_text → clean_text
    make_rag_text(text)                       : clean_text → RAG용 최종 텍스트

파이프라인:
    extract_pdf_text  →  clean_pdf_text  →  make_rag_text
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import fitz          # pymupdf
import pandas as pd


# ---------------------------------------------------------------------------
# (1) 원문 추출
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """PDF 파일에서 페이지별 텍스트를 추출해 단일 문자열로 반환."""
    page_texts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            page_texts.append(text)
    return "\n".join(page_texts).strip()


# ---------------------------------------------------------------------------
# (2) 정제 헬퍼 함수들
# ---------------------------------------------------------------------------

def build_spaced_phrase_pattern(phrase: str) -> str:
    """
    '제 안 요 청 서' 처럼 글자 사이에 공백이 삽입된 패턴을 감지하는 정규식 생성.
    단어 경계(lookbehind/lookahead)를 포함해 오탐을 줄인다.
    """
    chars = [re.escape(ch) for ch in phrase if ch.strip()]
    return r"(?<![가-힣A-Za-z0-9])" + r"\s*".join(chars) + r"(?![가-힣A-Za-z0-9])"


def fix_spaced_phrases(text: str, phrases: Optional[List[str]] = None) -> str:
    """
    문서 내 핵심 용어의 비정상 띄어쓰기를 복원한다.
    예) '제 안 요 청 서' → '제안요청서'
    """
    if phrases is None:
        phrases = [
            "제안요청서", "사업명", "사업비", "사업기간", "사업범위", "사업개요",
            "주관기관", "제안안내", "제안서", "제안요청", "통합시스템",
            "입학처", "서울시립대학교", "학업성취도", "종단분석", "추진배경",
            "필요성", "요구사항", "상세설명", "세부내용", "유지보수",
            "보안요구사항", "성능요구사항", "인터페이스", "데이터베이스",
            "사용자", "관리자", "클라우드", "인프라", "프로파일링",
        ]
    for phrase in phrases:
        text = re.sub(build_spaced_phrase_pattern(phrase), phrase, text)
    return text


def strip_toc_and_noise_lines(text: str) -> str:
    """
    목차·페이지 번호·반복 헤더 같은 전형적인 PDF 노이즈 라인을 제거한다.
    '목차' 섹션 내부의 목차 항목들도 제거한다.
    """
    cleaned_lines = []
    in_front_toc = False
    pending_toc_title = False

    for line in text.split("\n"):
        stripped = line.strip()
        compact = re.sub(r"\s+", "", stripped)

        if compact == "목차":
            in_front_toc = True
            pending_toc_title = False
            continue

        if stripped == "목":
            pending_toc_title = True
            continue

        if pending_toc_title and stripped == "차":
            in_front_toc = True
            pending_toc_title = False
            continue

        pending_toc_title = False

        if in_front_toc:
            toc_like = (
                re.search(r"[\.·]{4,}\s*\d{1,3}$", stripped) is not None
                or re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[\.)]?\s*[가-힣A-Za-z]", stripped) is not None
                or re.match(r"^\d+\s*[\.)]\s*[가-힣A-Za-z]", stripped) is not None
                or compact in {"l", "I"}
            )
            if not stripped or toc_like:
                continue
            if re.fullmatch(r"-\s*\d{1,3}\s*-", stripped):
                in_front_toc = False
                continue
            in_front_toc = False

        if not stripped:
            cleaned_lines.append("")
            continue

        if compact in {"[사전공개용]", "l", "I"}:
            continue
        if re.fullmatch(r"\d{1,3}", compact):
            continue
        if re.fullmatch(r"-\s*\d{1,3}\s*-", stripped):
            continue
        if re.fullmatch(r"페\s*이\s*지\s*:?\s*\d+\s*/\s*\d+", stripped):
            continue
        if re.fullmatch(r"(문\s*서\s*번\s*호|개\s*정\s*번\s*호|발\s*행\s*일)\s*:?", stripped):
            continue
        if re.search(r"[\.·]{4,}\s*\d{1,3}$", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_additional_phrases(text: str) -> str:
    """자주 발생하는 깨짐 표현을 사전 치환 방식으로 복원한다."""
    replacements = {
        "기개 발": "기개발",
        "시 스템": "시스템",
        "데 이터": "데이터",
        "클 라우드": "클라우드",
        "인 프라": "인프라",
        "상 세 설명": "상세설명",
        "세 부 내용": "세부내용",
        "사 업 비": "사업비",
        "사 업 명": "사업명",
        "사 업 기 간": "사업기간",
        "사 업 범 위": "사업범위",
        "주 관 기 관": "주관기관",
        "제 안 서": "제안서",
        "제 안 안 내": "제안안내",
        "요 구 사 항": "요구사항",
        "성 능 요 구 사 항": "성능요구사항",
        "보 안 요 구 사 항": "보안요구사항",
        "데 이 터 베 이 스": "데이터베이스",
        "프 로 파 일 링": "프로파일링",
        "용 역 업 체": "용역업체",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def truncate_appendix_sections(text: str) -> str:
    """
    붙임/별첨/서식/입찰안내 등 본문 이후 부록 구간을 잘라낸다.
    RAG 검색에서 부록 노이즈를 줄이기 위해 전체 길이의 35% 이후 지점부터 탐색한다.
    """
    cutoff_patterns = [
        r"\[\s*붙임\s*\d+\s*\]",
        r"\[\s*별첨\s*\d*\s*\]",
        r"\[\s*별지서식\s*\d*\s*\]",
        r"\[\s*서식\s*\d+\s*\]",
        r"별첨\s*[IVX]+\s*제안서\s*관련\s*서식",
        r"제안서\s*관련\s*서식",
        r"서식\s*\d+\s*:",
        r"제\s*[IVX]+\s*장\s*제안서\s*제출안내",
        r"제\s*[IVX]+\s*장\s*제안서\s*작성기준",
        r"제\s*[IVX]+\s*장\s*입찰\s*안내",
        r"입찰참가신청서",
        r"입찰\s*참가등록",
        r"입찰\s*참가\s*자격",
        r"입찰서류\s*및\s*제안서\s*제출",
        r"제안서\s*제출안내",
        r"제출서류",
        r"제안서\s*작성기준",
        r"제안서\s*작성방법",
        r"제안서\s*작성지침",
        r"제안서\s*작성지침\s*및\s*유의사항",
        r"입찰안내\s*사항",
        r"제안서\s*평가\s*및\s*협상",
        r"제안사\s*유의사항",
        r"입찰\s*공고문",
        r"입찰에\s*참가하고자\s*하는\s*자",
        r"청렴계약\s*서약서",
        r"안전보건관리\s*준수서약서",
        r"정량평가(?:기준|지표)",
        r"제안요구사항\s*수용\s*조견표",
        r"하도급계약\s*적정성",
        r"자가진단표",
    ]

    cutoff_idx = None
    min_start = int(len(text) * 0.35)

    for pattern in cutoff_patterns:
        for match in re.finditer(pattern, text):
            if match.start() >= min_start:
                cutoff_idx = match.start() if cutoff_idx is None else min(cutoff_idx, match.start())
                break

    if cutoff_idx is not None:
        text = text[:cutoff_idx]
    return text


def remove_contact_and_signature_noise(text: str) -> str:
    """서명란·날인 등 메타 정보를 제거한다."""
    return re.sub(r"\(인\)|서명|날인", " ", text)


# ---------------------------------------------------------------------------
# (3) 메인 정제 함수
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """
    raw_text에 정규화·노이즈 제거·표현 복원을 순차 적용해 clean_text를 생성한다.

    처리 순서:
        1. 유니코드 정규화 (NFKC) + 제어문자 제거
        2. 불필요 특수문자 제거
        3. 페이지 번호·문서 메타 헤더 제거
        4. 목차·반복 헤더 제거 (strip_toc_and_noise_lines)
        5. 항목 로마자 헤더 제거
        6. 페이지 구분선 제거
        7. 띄어쓰기 복원 (fix_spaced_phrases, fix_additional_phrases)
        8. 부록 구간 절단 (truncate_appendix_sections)
        9. 서명란 노이즈 제거
        10. 공백/줄바꿈 정리
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[  ]{2,}", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s\.\(\)\[\]\/,%:\-·?!@~&+*'\"=<>_]", " ", text)
    text = re.sub(r"페\s*이\s*지\s*:\s*\n?\s*\d+\s*/\s*\d+", " ", text)
    text = re.sub(r"(문\s*서\s*번\s*호|개\s*정\s*번\s*호|발\s*행\s*일)\s*:\s*\n?\s*[-\d\. ]+", " ", text)
    text = strip_toc_and_noise_lines(text)
    text = re.sub(r"(?ms)^(?:[IVX]+\.\s*[^\n]+\n\s*){2,8}", "", text, count=1)
    text = re.sub(r"(?m)^\s*[IVX]+\.\s*(?:제안|프로젝트|자격|수행능력|계약|기타|현황|개요|부문)[^\n]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*[-−–]\s*\d{1,4}\s*[-−–][ \t]*$", " ", text)
    text = re.sub(r"(?<!\d)[-−–]\s*\d{1,4}\s*[-−–](?!\d)", " ", text)
    text = re.sub(r"[\.·]{4,}", " ", text)
    text = re.sub(r"[-_=]{3,}", " ", text)
    text = fix_spaced_phrases(text)
    text = fix_additional_phrases(text)
    text = truncate_appendix_sections(text)
    text = remove_contact_and_signature_noise(text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"(?<![\.!?])\n(?!\n)", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def make_rag_text(text: str) -> str:
    """
    clean_text에서 RAG 검색에 적합한 최종 텍스트(rag_text)를 생성한다.

    clean_pdf_text 이후 추가로:
    - 붙임·별첨·서식·입찰안내 구간을 전체 길이의 25% 이후부터 탐색해 절단
    - 서명·날인 표현 제거
    - 연속 공백·빈 줄 정리
    """
    hard_cut_patterns = [
        r"\[\s*붙임\s*\d+\s*\]",
        r"\[\s*별첨\s*\d*\s*\]",
        r"\[\s*별지서식\s*\d*\s*\]",
        r"\[\s*서식\s*\d+\s*\]",
        r"제\s*[IVX]+\s*장\s*제안서\s*제출안내",
        r"제\s*[IVX]+\s*장\s*제안서\s*작성기준",
        r"제\s*[IVX]+\s*장\s*입찰\s*안내",
        r"입찰\s*참가등록",
        r"입찰\s*참가\s*자격",
        r"입찰서류\s*및\s*제안서\s*제출",
        r"제안서\s*제출안내",
        r"제출서류",
        r"제안서\s*작성기준",
        r"제안서\s*작성방법",
        r"제안서\s*작성지침",
        r"제안서\s*평가\s*및\s*협상",
        r"제안사\s*유의사항",
        r"입찰안내\s*사항",
        r"입찰\s*공고문",
        r"입찰에\s*참가하고자\s*하는\s*자",
        r"정량평가(?:기준|지표)",
        r"제안요구사항\s*수용\s*조견표",
        r"하도급계약\s*적정성",
        r"자가진단표",
    ]

    cutoff_idx = None
    min_start = int(len(text) * 0.25)

    for pattern in hard_cut_patterns:
        for match in re.finditer(pattern, text):
            if match.start() >= min_start:
                cutoff_idx = match.start() if cutoff_idx is None else min(cutoff_idx, match.start())
                break

    if cutoff_idx is not None:
        text = text[:cutoff_idx]

    text = re.sub(r"\(인\)|서명|날인", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# (4) 폴더 단위 파싱 엔트리포인트
# ---------------------------------------------------------------------------

def parse_pdf_folder(
    folder_path: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    폴더 내 모든 .pdf 파일을 파싱해 DataFrame을 반환한다.

    Args:
        folder_path : PDF 파일들이 있는 디렉토리 경로.
        output_dir  : 지정 시 df_parsed_pdf_hard.csv 를 해당 디렉토리에 저장한다.

    Returns:
        columns = ["파일명", "파일형식", "raw_text", "clean_text", "rag_text"]
    """
    folder = Path(folder_path)
    pdf_files = sorted([p for p in folder.glob("*.pdf") if p.is_file()])

    print(f"대상 PDF 파일 수: {len(pdf_files)}")

    rows = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        raw_text = extract_pdf_text(pdf_path)
        clean_text = clean_pdf_text(raw_text)
        rag_text = make_rag_text(clean_text)
        rows.append({
            "파일명": pdf_path.name,
            "파일형식": "pdf",
            "raw_text": raw_text,
            "clean_text": clean_text,
            "rag_text": rag_text,
        })

    df = pd.DataFrame(rows, columns=["파일명", "파일형식", "raw_text", "clean_text", "rag_text"])

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "df_parsed_pdf_hard.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"저장 완료: {csv_path}  ({len(df)}건)")

    return df
