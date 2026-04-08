from langchain_openai import OpenAIEmbeddings
import config

def get_embedding_model():
    """
    프로젝트 전역에서 사용할 임베딩 모델을 호출합니다.
    """
    model_name = config.EMBED_MODEL
    print(f"임베딩 모델 로드: {model_name}")
    
    return OpenAIEmbeddings(model=model_name)