from __future__ import annotations

import sys
from pathlib import Path
import fitz

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd

# ─── 프로젝트 루트를 sys.path 에 추가 (cli.py 와 동일한 import 경로 확보) ───
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_api_v4 import get_rag_context
from src.generation.generator import BidCoinGenerator
from src.generation.schemas import RetrievalResult

# ─── 페이지 설정  ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Bidcoin RAG", layout="wide")

# ─── 상수 ──────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 5          # cli.py 와 동일
PDF_OUTPUT_DIR    = Path("/home/bidcoin/output_pdf")
DATA_LIST_PATH    = "/home/bidcoin/data_list_metadata.csv"


# ─── 데이터 로딩 (최초 1회) ────────────────────────────────────────────────
@st.cache_data
def load_file_list():
    df = pd.read_csv(DATA_LIST_PATH, encoding="utf-8")
    file_list     = list(df["파일명"])
    file_type_map = {df["파일명"][i]: df["파일형식"][i] for i in range(len(df))}
    return file_list, file_type_map


try:
    ls, dc = load_file_list()
except Exception as _e:
    ls, dc = [], {}
    st.warning(f"파일 목록 로드 실패: {_e}")


# ─── Generator 캐시 (모델 초기화 비용 1회만) ──────────────────────────────
@st.cache_resource
def get_generator() -> BidCoinGenerator:
    return BidCoinGenerator()


# ─── cli.py 와 동일한 히스토리 트리밍 ─────────────────────────────────────
def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """최근 max_turns 턴(user+assistant 쌍)만 남긴다."""
    return history[-(max_turns * 2):]


# ─── 실제 RAG 파이프라인 호출 (cli.py 의 main() 로직을 함수로 추출) ────────
def run_rag(query: str, chat_history: list[dict]) -> tuple[str, list[str], str]:
    """
    Returns
    -------
    answer       : LLM 생성 답변
    used_sources : 사용된 출처 파일명 리스트
    status       : "CHITCHAT" | "SEARCH_SUCCESS" | "NO_INFO"
    """
    recent_history = trim_history(chat_history)
    raw_result     = get_rag_context(query, recent_history)
    status         = raw_result.get("status", "NO_INFO")

    retrieval_result = RetrievalResult.model_validate(raw_result)
    generator        = get_generator()
    result           = generator.generate(retrieval_result)

    return result.answer, result.used_sources, status


# ══════════════════════════════════════════════════════════════════════════════
# 페이지 ① — LLM 채팅
# ══════════════════════════════════════════════════════════════════════════════
def page_llm():
    st.title("Bidcoin")
    st.caption("RAG demo developed by BidMind")

    left, right = st.columns([1, 2.4], gap="large")

    # ── 왼쪽: 설정 패널 ──────────────────────────────────────────────────────
    with left:
        st.subheader("LLM 설정")

        if "selected_llm" not in st.session_state:
            st.session_state.selected_llm = "Chatgpt"

        llm_name = st.selectbox(
            "모델 선택",
            ["Chatgpt", "Gemini", "Claude", "Gemma", "Llama3", "Qwen"],
            key="selected_llm",
        )

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.1

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            key="temperature",
        )
        temperature = st.session_state.temperature

        if 0.1 <= temperature <= 0.5:
            color, status_text = "#22c55e", "권장 범위입니다"
        else:
            color, status_text = "#ef4444", "권장 범위를 벗어났습니다"

        st.markdown(
            f"""
            <div style="
                background:#111827; padding:16px; border-radius:12px;
                border:1px solid #2a2f3a; margin-top:12px;
            ">
                <div style="font-size:14px; color:#9ca3af; margin-bottom:8px;">현재 설정</div>
                <div style="font-size:16px; font-weight:700; margin-bottom:8px;">
                    선택한 모델: {llm_name}
                </div>
                <div style="font-size:18px; font-weight:800; color:{color};">
                    Temperature: {temperature:.1f}
                </div>
                <div style="font-size:14px; color:{color}; margin-top:6px;">{status_text}</div>
                <div style="font-size:13px; color:#cbd5e1; margin-top:8px;">
                    추천 Temperature: 0.1 ~ 0.5
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 설명")
        st.write("- ChatGPT-5 model은 Temperature 설정을 무시함")
        st.write("- 낮을수록 더 안정적이고 일관된 답변")
        st.write("- 높을수록 더 다양한 답변")
        st.write("- RAG QA는 보통 0.1 ~ 0.5 추천")

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.raw_history = []
            st.rerun()

    # ── 오른쪽: 채팅 패널 ────────────────────────────────────────────────────
    with right:
        st.subheader("질문하기")

        # 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []          # 화면 표시용
        if "raw_history" not in st.session_state:
            st.session_state.raw_history = []       # RAG API 전달용 (role/content 형식)

        chat_box = st.container(height=600)

        with chat_box:
            if not st.session_state.messages:
                st.markdown(
                    """
                    <div style="
                        padding:24px; border:1px solid #2a2f3a; border-radius:16px;
                        background:#0f172a; color:#cbd5e1;
                    ">
                        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">
                            무엇을 도와드릴까요?
                        </div>
                        <div style="font-size:14px; color:#94a3b8;">
                            예시 질문:
                            <ul style="margin-top:4px;">
                                <li>한영대학교 트랙운영 학사정보시스템 고도화 사업의 주요 목적은 무엇인가?</li>
                                <li>EIP3.0 고압가스 안전관리 시스템 구축 용역의 발주 기관과 예산은?</li>
                                <li>스포츠윤리센터 LMS 기능개선 사업의 예산과 공고 일자는?</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    # 출처가 있으면 assistant 버블 아래에 표시
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("사용된 출처"):
                            for src in msg["sources"]:
                                st.markdown(f"- `{src}`")

                    # 파이프라인 상태 뱃지
                    if msg["role"] == "assistant" and "status" in msg:
                        _badge_color = {
                            "SEARCH_SUCCESS": "#22c55e",
                            "CHITCHAT":       "#60a5fa",
                            "NO_INFO":        "#f59e0b",
                        }.get(msg["status"], "#9ca3af")
                        st.markdown(
                            f'<span style="font-size:11px; color:{_badge_color};">'
                            f'● {msg["status"]}</span>',
                            unsafe_allow_html=True,
                        )

        prompt = st.chat_input("질문을 입력하세요")

        if prompt:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 실제 RAG 파이프라인 실행
            with st.spinner("RAG 파이프라인 실행 중..."):
                try:
                    answer, used_sources, status = run_rag(
                        query=prompt,
                        chat_history=st.session_state.raw_history,
                    )
                except Exception as exc:
                    answer       = f"오류가 발생했습니다: {type(exc).__name__}: {exc}"
                    used_sources = []
                    status       = "ERROR"

            # 어시스턴트 메시지 (출처·상태 포함) 추가
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "sources": used_sources,
                "status":  status,
            })

            # raw_history 업데이트 (cli.py 와 동일한 구조)
            st.session_state.raw_history.append({"role": "user",      "content": prompt})
            st.session_state.raw_history.append({"role": "assistant", "content": answer})

            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 페이지 ② — 파일 뷰어
# ══════════════════════════════════════════════════════════════════════════════
def page_files():
    st.title("파일 뷰어")
    files = list(ls)
    left, right = st.columns([1, 2.4], gap="large")

    if "viewer_page"   not in st.session_state: st.session_state.viewer_page   = 1
    if "current_file"  not in st.session_state: st.session_state.current_file  = None
    if "search_pages"  not in st.session_state: st.session_state.search_pages  = []
    if "search_idx"    not in st.session_state: st.session_state.search_idx    = 0
    if "last_keyword"  not in st.session_state: st.session_state.last_keyword  = ""

    with left:
        selected_file = st.selectbox("파일 선택", files)
        st.write("선택한 파일:", selected_file)

        pdf_name = f"{Path(selected_file).stem}.pdf"
        pdf_path = PDF_OUTPUT_DIR / pdf_name

        if st.session_state.current_file != selected_file:
            st.session_state.current_file = selected_file
            st.session_state.viewer_page  = 1
            st.session_state.search_pages = []
            st.session_state.search_idx   = 0
            st.session_state.last_keyword = ""

        keyword = st.text_input("문서 검색", placeholder="검색어 입력")

        def find_text_pages(pdf_path, keyword):
            pages = []
            doc   = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                if page.search_for(keyword):
                    pages.append(i + 1)
            doc.close()
            return pages

        search_col1, search_col2 = st.columns([1, 1])

        with search_col1:
            if st.button("검색", use_container_width=True):
                if pdf_path.exists() and keyword.strip():
                    pages = find_text_pages(str(pdf_path), keyword.strip())
                    st.session_state.search_pages = pages
                    st.session_state.search_idx   = 0
                    st.session_state.last_keyword = keyword.strip()
                    st.session_state.viewer_page  = pages[0] if pages else 1
                    st.rerun()

        with search_col2:
            if st.button("검색 초기화", use_container_width=True):
                st.session_state.search_pages = []
                st.session_state.search_idx   = 0
                st.session_state.last_keyword = ""
                st.rerun()

        if st.session_state.search_pages:
            current_idx  = st.session_state.search_idx
            current_page = st.session_state.search_pages[current_idx]

            st.success(
                f"검색 결과: {st.session_state.search_pages}\n\n"
                f"현재 검색 위치: {current_idx + 1} / {len(st.session_state.search_pages)}"
                f" → {current_page}페이지"
            )

            r1, r2 = st.columns([1, 1])
            with r1:
                if st.button("이전 검색 결과", use_container_width=True):
                    st.session_state.search_idx   = max(0, st.session_state.search_idx - 1)
                    st.session_state.viewer_page  = st.session_state.search_pages[st.session_state.search_idx]
                    st.rerun()
            with r2:
                if st.button("다음 검색 결과", use_container_width=True):
                    st.session_state.search_idx = min(
                        len(st.session_state.search_pages) - 1,
                        st.session_state.search_idx + 1,
                    )
                    st.session_state.viewer_page = st.session_state.search_pages[st.session_state.search_idx]
                    st.rerun()

        st.markdown("### 페이지 이동")
        nav1, nav2, nav3 = st.columns([1, 1.2, 1])

        with nav1:
            if st.button("◀ 이전", use_container_width=True):
                st.session_state.viewer_page = max(1, st.session_state.viewer_page - 1)
                st.rerun()
        with nav2:
            page_input = st.number_input("페이지", min_value=1, step=1, value=st.session_state.viewer_page)
            if page_input != st.session_state.viewer_page:
                st.session_state.viewer_page = page_input
                st.rerun()
        with nav3:
            if st.button("다음 ▶", use_container_width=True):
                st.session_state.viewer_page += 1
                st.rerun()

        if st.session_state.search_pages:
            selected_result_page = st.selectbox("검색 결과 페이지 선택", st.session_state.search_pages)
            if st.button("선택 페이지로 이동", use_container_width=True):
                st.session_state.viewer_page = selected_result_page
                st.session_state.search_idx  = st.session_state.search_pages.index(selected_result_page)
                st.rerun()

    with right:
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_viewer(
                input=pdf_bytes,
                width="100%",
                height=900,
                render_text=True,
                scroll_to_page=st.session_state.viewer_page,
            )
        else:
            st.warning(f"아직 PDF가 없습니다: {pdf_name}")


# ══════════════════════════════════════════════════════════════════════════════
# 라우팅
# ══════════════════════════════════════════════════════════════════════════════
menu = st.sidebar.selectbox("메뉴", ["LLM", "파일뷰어"])

if menu == "LLM":
    page_llm()
elif menu == "파일뷰어":
    page_files()
