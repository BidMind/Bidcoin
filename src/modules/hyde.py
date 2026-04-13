# 검색 명중률을 극한으로 끌어올리기 위해 가짜 입찰 문서를 생성합니다.
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_hyde_document(query: str) -> str:
    """질문에 대한 가상의 제안요청서(RFP) 단락을 생성합니다."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 공공기관 입찰 문서 작성자야. 
        사용자의 질문에 대한 가상의 정답을 '실제 입찰 제안요청서(RFP)에 적혀있을 법한 공식적이고 전문적인 문어체'로 1문단(약 200자) 작성해.
        숫자나 고유명사가 틀려도 상관없으니 구조와 어휘를 입찰 문서처럼 흉내 내는 것에 집중해."""),
        ("human", "{query}")
    ])
    
    try:
        return (prompt | llm).invoke({"query": query}).content.strip()
    except Exception as e:
        print(f"⚠️ [HyDE 에러] 원본 질문으로 대체: {e}")
        return query