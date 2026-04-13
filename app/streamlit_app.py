from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import fitz
import sys
import importlib.util
from streamlit_pdf_viewer import pdf_viewer


"""
다음 구현 기능:

1. 데이터 전처리 - > Parsing -> Chunking -> 임베딩 생성 -> 벡터 DB 구축 작업 버튼 하나

2. 세션 나눠서 대화 진행


"""

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd

df = pd.read_csv("/home/bidcoin/data_list_metadata.csv", encoding="utf-8") 
ls = list(df["파일명"])
dc = {df["파일명"][i]: df["파일형식"][i] for i in range(len(df))}  # 파일명-파일경로 매핑 딕셔너리


# 더미 RAG 함수
# -----------------------------
def run_rag(query, llm_name, temperature):
    return f"[{llm_name} | temp={temperature:.1f}] 질문 '{query}'에 대한 예시 답변입니다."

# 세션 상태 초기화
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def page_llm():
    st.set_page_config(page_title="RAG UI", layout="wide")
    st.title("Bidcoin")
    st.caption("RAG demo developed by BidMind")

    left, right = st.columns([1, 2.4], gap="large")

    # 왼쪽: 설정 패널
    with left:
        st.subheader("LLM 설정")

        # 모델 선택 드롭다운
        if "selected_llm" not in st.session_state:
            st.session_state.selected_llm = "Chatgpt"

        llm_name = st.selectbox(
            "모델 선택",
            ["Chatgpt", "Gemini", "Claude", "Gemma", "Llama3", "Qwen"],
            key="selected_llm"
        )

        # Temperature 슬라이더
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.1

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            key="temperature"
        )

        temperature = st.session_state.temperature

        if 0.1 <= temperature <= 0.5:
            color = "#22c55e"
            status_text = "권장 범위입니다"
        else:
            color = "#ef4444"
            status_text = "권장 범위를 벗어났습니다"

        st.markdown(
            f"""
            <div style="
                background:#111827;
                padding:16px;
                border-radius:12px;
                border:1px solid #2a2f3a;
                margin-top:12px;
            ">
                <div style="font-size:14px; color:#9ca3af; margin-bottom:8px;">현재 설정</div>
                <div style="font-size:16px; font-weight:700; margin-bottom:8px;">
                    선택한 모델: {llm_name}
                </div>
                <div style="font-size:18px; font-weight:800; color:{color};">
                    Temperature: {temperature:.1f}
                </div>
                <div style="font-size:14px; color:{color}; margin-top:6px;">
                    {status_text}
                </div>
                <div style="font-size:13px; color:#cbd5e1; margin-top:8px;">
                    추천 Temperature: 0.1 ~ 0.5
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 설명")
        st.write("- 낮을수록 더 안정적이고 일관된 답변")
        st.write("- 높을수록 더 다양한 답변")
        st.write("- RAG QA는 보통 0.1 ~ 0.5 추천")

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 오른쪽: 질문/답변 패널
    with right:
        st.subheader("질문하기")

        chat_box = st.container(height=600)

        with chat_box:
            if not st.session_state.messages:
                st.markdown(
                    """
                    <div style="
                        padding:24px;
                        border:1px solid #2a2f3a;
                        border-radius:16px;
                        background:#0f172a;
                        color:#cbd5e1;
                    ">
                        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">무엇을 도와드릴까요?</div>
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
                    unsafe_allow_html=True
                )

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("질문을 입력하세요")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            answer = run_rag(
                query=prompt,
                llm_name=st.session_state.selected_llm,
                temperature=st.session_state.temperature
            )

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


def page_files():
    st.set_page_config(page_title="문서 뷰어", layout="wide")
    PDF_OUTPUT_DIR = Path("/home/bidcoin/output_pdf")
    st.title("파일 뷰어")
    files = [i for i in ls]
    left, right = st.columns([1, 2.4], gap="large")
    
    if "viewer_page" not in st.session_state:
        st.session_state.viewer_page = 1

    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    if "search_pages" not in st.session_state:
        st.session_state.search_pages = []

    if "search_idx" not in st.session_state:
        st.session_state.search_idx = 0

    if "last_keyword" not in st.session_state:
        st.session_state.last_keyword = ""

    with left:
        selected_file = st.selectbox("파일 선택", files)
        st.write("선택한 파일:", selected_file)

        pdf_name = f"{Path(selected_file).stem}.pdf"
        pdf_path = PDF_OUTPUT_DIR / pdf_name

        if st.session_state.current_file != selected_file:
            st.session_state.current_file = selected_file
            st.session_state.viewer_page = 1
            st.session_state.search_pages = []
            st.session_state.search_idx = 0
            st.session_state.last_keyword = ""

        keyword = st.text_input("문서 검색", placeholder="검색어 입력")


        def find_text_pages(pdf_path, keyword):
            pages = []
            doc = fitz.open(pdf_path)


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
                    st.session_state.search_idx = 0
                    st.session_state.last_keyword = keyword.strip()

                    if pages:
                        st.session_state.viewer_page = pages[0]
                    else:
                        st.session_state.viewer_page = 1
                    st.rerun()

        with search_col2:
            if st.button("검색 초기화", use_container_width=True):
                st.session_state.search_pages = []
                st.session_state.search_idx = 0
                st.session_state.last_keyword = ""
                st.rerun()

        if st.session_state.search_pages:
            current_idx = st.session_state.search_idx
            current_page = st.session_state.search_pages[current_idx]

            st.success(
                f"검색 결과: {st.session_state.search_pages}\n\n"
                f"현재 검색 위치: {current_idx + 1} / {len(st.session_state.search_pages)}"
                f" → {current_page}페이지"
            )

            r1, r2 = st.columns([1, 1])

            with r1:
                if st.button("이전 검색 결과", use_container_width=True):
                    st.session_state.search_idx = max(0, st.session_state.search_idx - 1)
                    st.session_state.viewer_page = st.session_state.search_pages[st.session_state.search_idx]
                    st.rerun()

            with r2:
                if st.button("다음 검색 결과", use_container_width=True):
                    st.session_state.search_idx = min(
                        len(st.session_state.search_pages) - 1,
                        st.session_state.search_idx + 1
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
            page_input = st.number_input(
                "페이지",
                min_value=1,
                step=1,
                value=st.session_state.viewer_page
            )
            if page_input != st.session_state.viewer_page:
                st.session_state.viewer_page = page_input
                st.rerun()

        with nav3:
            if st.button("다음 ▶", use_container_width=True):
                st.session_state.viewer_page += 1
                st.rerun()
                
        if st.session_state.search_pages:
            selected_result_page = st.selectbox(
                "검색 결과 페이지 선택",
                st.session_state.search_pages
            )
            if st.button("선택 페이지로 이동", use_container_width=True):
                st.session_state.viewer_page = selected_result_page
                st.session_state.search_idx = st.session_state.search_pages.index(selected_result_page)
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
            scroll_to_page=st.session_state.viewer_page
        )
        else:
            st.warning(f"아직 PDF가 없습니다: {pdf_name}")


menu = st.sidebar.selectbox("메뉴", ["LLM", "파일뷰어"])

if menu == "LLM":
    page_llm()
elif menu == "파일뷰어":
    page_files()



#st.text("그냥 텍스트 출력")
#st.markdown("**마크다운** 지원합니다.")
#st.write("텍스트, 숫자, DataFrame 등 거의 다 자동 렌더링")