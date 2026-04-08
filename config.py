import os
import sys

# 현재 파일이 있는 최상단 폴더를 파이썬 경로에 강제 추가, import 오류 방지
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# 입출력 파일 경로 설정
CSV_PATH = os.path.join(ROOT_DIR, "processed_data.csv")
FAISS_INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")

# 모델 이름 설정
EMBED_MODEL = "text-embedding-3-small"    
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
