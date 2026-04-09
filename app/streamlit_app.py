from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
import importlib.util


#ROOT_DIR = Path(__file__).resolve().parent.parent
#if str(ROOT_DIR) not in sys.path:
    #sys.path.insert(0, str(ROOT_DIR))

#from src.preprocessing.metadata_cleaning import process_metadata

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd

#df = pd.read_csv("/home/shared/data_list.csv", encoding="utf-8") 
#df = process_metadata(df, files_dir="/home/shared/files")

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
    st.title("파일 뷰어")
    files = ["a", "b", "c"]
    selected_file = st.selectbox("파일 선택", files)
    st.write("선택한 파일:", selected_file)


menu = st.sidebar.selectbox("메뉴", ["LLM", "파일뷰어"])

if menu == "LLM":
    page_llm()
elif menu == "파일뷰어":
    page_files()



#st.text("그냥 텍스트 출력")
#st.markdown("**마크다운** 지원합니다.")
#st.write("텍스트, 숫자, DataFrame 등 거의 다 자동 렌더링")