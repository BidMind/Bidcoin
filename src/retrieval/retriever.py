import pandas as pd
import config
import re
from langchain_core.documents import Document
from src.embedding.vector_store import load_db
from src.modules.hyde import is_field_query, FIELD_QUERY_KEYWORDS
from typing import Optional
from config import OUTPUT_DIR

# def retrieve_candidates(query: str, k: int = 10):
#     """
#     [기능] 빠르지만 정확도가 떨어지는 1차 검색을 수행합니다.
#     [흐름] 질문(query) -> 벡터 DB -> DB에서 가장 유사한 k개 문서 반환
#     """
#     vector_store = load_db()
#     # similarity_search: 벡터 DB에서 질문과 가장 유사한 k개의 문서를 반환
#     candidates = vector_store.similarity_search(query, k=k)
#     return candidates


# 수정: alias 확장 + 메타데이터 직접 조회 추가+ 실패 시 dense 1회 fallback -------
# ============================================================
# 1) Alias 확장 사전
# ============================================================
# alias 확장 사전 (축약 → 정식명칭)
ALIAS_MAP = {
    # 대학교 (education)
    "경희대": "경희대학교",
    "고려대": "고려대학교",
    "광주과기원": "광주과학기술원",
    "남서울대": "남서울대학교",
    "대전대": "대전대학교",
    "서영대": "서영대학교 산학협력단",
    "서울시립대": "서울시립대학교",
    "을지대": "을지대학교",
    "전북대": "전북대학교",
    "조선대": "조선대학교",

    # 공공기관 (public)
    "수자원공사": "한국수자원공사",
    "철도공사": "한국철도공사 (용역)",
    "코레일": "한국철도공사 (용역)",
    "농어촌공사": "한국농어촌공사",
    "가스공사": "한국가스공사",
    "수출입은행": "한국수출입은행",
    "국민연금": "국민연금공단",
    "산업인력공단": "한국산업인력공단",
    "산업단지공단": "한국산업단지공단",
    "전기안전공사": "한국전기안전공사",
    "농수산공사": "한국농수산식품유통공사",
    "국가철도공단": "국가철도공단",
    "고양도시공사": "고양도시관리공사",
    "부산관광": "부산관광공사",
    "파주도시공사": "파주도시관광공사",
    "KOICA": "KOICA 전자조달",
    "코이카": "KOICA 전자조달",

    # 연구기관 (research)
    "생산기술연구원": "한국생산기술연구원",
    "연구재단": "한국연구재단",
    "원자력연구원": "한국원자력연구원",
    "한의학연구원": "한국한의학연구원",
    "충북연구원": "재단법인충북연구원",
    "광주연구원": "재단법인 광주연구원",

    # 비영리/협회 (nonprofit)
    "수협": "수협중앙회",
    "벤처협회": "(사)벤처기업협의회",
    "보험개발원": "사단법인 보험개발원",
    "부산영화제": "(사)부산국제영화제",

    # 중앙/지방 (central/local)
    "선관위": "중앙선거관리위원회",
    "서울교육청": "서울특별시교육청",
}

def expand_alias(query: str) -> str:
    """축약 기관명을 정식 명칭으로 확장"""
    expanded = query
    for alias, full in ALIAS_MAP.items():
        if alias in expanded:
            expanded = expanded.replace(alias, full)

    if expanded != query:
        print(f"[Alias 확장] '{query}' → '{expanded}'")
    return expanded


# ============================================================
# 2) 메타데이터 직접 조회
#    검색용: source, 발주 기관, 사업명
#    출력용: source, 발주 기관, 사업명, 사업 금액, 입찰 참여 마감일
# ============================================================
def lookup_metadata(expanded_query: str) -> list[Document]:
    try:
        df = pd.read_csv(OUTPUT_DIR / "data_list_metadata.csv", encoding="utf-8").fillna("")
    except Exception as e:
        print(f"[메타 조회] CSV 로드 실패 → dense fallback: {e}")
        return []

    # alias 확장된 쿼리 전체를 그대로 contains 검색
    search_cols = [col for col in ["파일명", "발주 기관", "사업명"] if col in df.columns]

    mask = None
    for col in search_cols:
        col_mask = df[col].astype(str).str.contains(expanded_query, na=False, regex=False)
        mask = col_mask if mask is None else (mask | col_mask)

    # 매칭 안 되면 단어 단위로 재시도
    if mask is None or not mask.any():
        words = [w for w in expanded_query.split() if len(w) > 1]
        for word in words:
            for col in search_cols:
                col_mask = df[col].astype(str).str.contains(word, na=False, regex=False)
                mask = col_mask if mask is None else (mask | col_mask)

    if mask is None or not mask.any():
        print("[메타 조회] 일치하는 기관/사업명 없음 → dense fallback")
        return []

    matched = df[mask].copy()
    docs = []
    for _, row in matched.iterrows():
        src      = row.get("파일명", "")
        org      = row.get("발주 기관", "미상")
        biz      = row.get("사업명", "미상")
        amount   = row.get("사업 금액", "정보 없음")
        start    = row.get("입찰 참여 시작일", "정보 없음")
        deadline = row.get("입찰 참여 마감일", "정보 없음")
        notice   = row.get("공고 번호", "정보 없음")
        rnd      = row.get("공고 차수", "정보 없음")

        content = (
            f"[발주기관: {org} | 사업명: {biz}]\n"
            f"공고 번호: {notice} | 공고 차수: {rnd}\n"
            f"사업 금액: {amount}\n"
            f"입찰 참여 시작일: {start}\n"
            f"입찰 참여 마감일: {deadline}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": src, "retrieval_type": "metadata_lookup"}
        ))

    print(f"[메타 조회] {len(docs)}건 직접 조회 성공")
    return docs


# ============================================================
# 3) 메인 retrieve 함수
#    필드조회형: metadata 먼저, 실패 시 dense 1회
#    일반 질문: HyDE + 원본쿼리 병행
# ============================================================
def retrieve_candidates(hyde_doc: str, original_query: str, k: int = 10):
    """
    - 필드조회형 질문: 메타데이터 직접 조회 우선, 부족하면 dense 보완
    - 일반 질문: HyDE + 원본쿼리 병행 dense 검색
    """
    expanded_query = expand_alias(original_query)
    vector_store = load_db()

    # 필드조회형 질문
    if is_field_query(original_query):
        meta_docs = lookup_metadata(expanded_query)
        if meta_docs:
            return meta_docs[:k]
        # 메타 조회 실패 시 dense fallback
        print("[Fallback] dense 검색으로 전환")
        return vector_store.similarity_search(expanded_query, k=k)

    # 일반 질문 : Dense 검색 (HyDE + 원본쿼리 병행)
    hyde_results  = vector_store.similarity_search(hyde_doc, k=k)
    query_results = vector_store.similarity_search(expanded_query, k=k)

    seen, merged = set(), []
    for doc in hyde_results + query_results:
        key = (
            doc.metadata.get("source", ""),
            doc.page_content[:200],
        )
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged[:k]