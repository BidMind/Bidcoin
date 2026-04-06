import os
from langchain_openai import OpenAIEmbeddings

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

def get_embedding_model():
    """
    프로젝트 전역에서 사용할 임베딩 모델을 호출합니다.
    """
    # Bid Coin 프로젝트 지정 모델
    model_name = "text-embedding-3-small"
    print(f"임베딩 모델 로드: {model_name}")
    
    return OpenAIEmbeddings(model=model_name)