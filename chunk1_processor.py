import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_and_chunk(raw_text, notice_id, agency, project_name, amt_val, start_date):
    """텍스트를 정제하고, 길이에 따라 청킹한 뒤 메타데이터를 주입합니다."""
    
    # 데이터가 없거나 빈 경우 빈 리스트 반환
    if pd.isna(raw_text) or len(str(raw_text).strip()) == 0:
        return []
    
    # ------------------------------------------------------------
    # [Step 1] 정보 보존형 텍스트 정제
    # ------------------------------------------------------------
    allowed_chars = r'[^가-힣a-zA-Z0-9\s\.\(\)\[\]\/\,\%\:\-\·\?\!\@\+\=\<\>\~\&\*\"\'\■\※\•\○\●\·\_\<\>\=\㎡\㎥\원]'
    cleaned_text = re.sub(allowed_chars, ' ', str(raw_text))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_len = len(cleaned_text)
    
    # ------------------------------------------------------------
    # [Step 2] 문장 맥락 보존형 청킹
    # ------------------------------------------------------------
    if cleaned_len < 500:
        chunk_size, chunk_overlap = 500, 0
    elif cleaned_len < 3000:
        chunk_size, chunk_overlap = 600, 100
    else:
        chunk_size, chunk_overlap = 1000, 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
    )
    chunks = splitter.split_text(cleaned_text)
    
    # ------------------------------------------------------------
    # [Step 3] 메타데이터 주입 및 안전한 문자열 처리
    # ------------------------------------------------------------
    def safe_str(val):
        return "미상" if pd.isna(val) or str(val).strip() == "" else str(val).strip()
    
    # 금액 포맷팅
    try:
        amt_str = f"{float(amt_val):,.0f}원" if not pd.isna(amt_val) else "미상"
    except (ValueError, TypeError):
        amt_str = safe_str(amt_val)
        
    notice_id_str = safe_str(notice_id)
    start_date_str = safe_str(start_date)
    agency_str = safe_str(agency)
    project_name_str = safe_str(project_name)
    
    total_chunks = len(chunks)
    chunks_with_meta = []
    
    for i, chunk_text in enumerate(chunks):
        meta_header = (
            f"[공고번호: {notice_id_str} | 발주기관: {agency_str} | "
            f"사업명: {project_name_str} | 금액: {amt_str} | "
            f"시작일: {start_date_str} | 조각순서: {i+1}/{total_chunks}]"
        )
        chunk_with_meta = f"{meta_header}\n\n{chunk_text}"
        chunks_with_meta.append(chunk_with_meta)
        
    return chunks_with_meta