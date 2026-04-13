# 사용자의 질문이 단순 대화인지, DB 검색이 필요한 질문인지 판별합니다.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def route_query(query: str) -> str:
    """질문의 의도를 파악하여 'CHAT' 또는 'SEARCH'로 라우팅합니다."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 질문 분류기야. 
        사용자의 질문이 단순 인사말, 칭찬, 감정 표현 등 검색이 필요 없는 일상 대화라면 'CHAT'을,
        입찰, 예산, 사업, 마감일, 규정 등 DB 검색이 필요한 정보성 질문이라면 'SEARCH'를 출력해.
        오직 'CHAT' 또는 'SEARCH' 단어 하나만 출력할 것."""),
        ("human", "{query}")
    ])
    
    try:
        decision = (prompt | llm).invoke({"query": query}).content.strip().upper()
        return decision if decision in ["CHAT", "SEARCH"] else "SEARCH"
    except Exception as e:
        print(f"⚠️ [라우터 에러] 기본값(SEARCH)으로 진행: {e}")
        return "SEARCH"