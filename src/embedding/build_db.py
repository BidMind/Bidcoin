# 벡터 DB 구축 및 로컬 저장 
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embedding import get_embedding_model  # 분리한 임베딩 모듈 호출

# 경로 설정 (실제 경로로 수정 필요)
BASE_DIR = r"D:\VS Test\bidcoin" # 실제 경로로 변경
CSV_PATH = os.path.join(BASE_DIR, "processed_data.csv") # 실제 파일명으로 변경
FAISS_DB_PATH = os.path.join(BASE_DIR, "faiss_index") # DB가 저장될 폴더명

def build_and_save_db():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 전처리 파일이 없습니다: {CSV_PATH}")
        return

    print("[DB 구축 모드] FAISS 벡터 DB 생성을 시작합니다...")
    
    # 1. 데이터 로드
    df = pd.read_csv(CSV_PATH)
    docs = [Document(page_content=str(row['content']), metadata={"source": row['source']}) for _, row in df.iterrows()]
    
    # 2. 전용 모듈에서 임베딩 모델 가져오기
    embeddings = get_embedding_model()
    
    # 3. FAISS 인덱스 생성
    print(f"{len(docs)}개 청크를 벡터로 변환 중... (약간의 시간이 소요됩니다)")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 4. 물리적 폴더에 DB 저장 (가장 중요!)
    vectorstore.save_local(FAISS_DB_PATH)
    print(f"벡터 DB 구축 및 로컬 저장 완료! -> {FAISS_DB_PATH}")

if __name__ == "__main__":
    build_and_save_db()