# 최종 LLM에게 넘길 문서를 다이어트하고, 쓸모없는 문서는 걸러냅니다.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def compress_document(query: str, text: str) -> str:
    """문서에서 질문과 관련된 핵심만 발췌하고, 없으면 'PASS'를 반환합니다."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문에 답하기 위해 꼭 필요한 핵심 팩트만 문서에서 발췌해.
        원문을 요약하지 말고 필요한 부분만 그대로 발췌해.
        문서에 질문과 관련된 내용이 전혀 없다면 오직 'PASS' 라고만 출력해."""),
        ("human", "[질문]: {query}\n\n[원본 문서]: {text}")
    ])
    
    try:
        return (prompt | llm).invoke({"query": query, "text": text}).content.strip()
    except Exception as e:
        print(f"⚠️ [압축 에러] 원본 문서 유지: {e}")
        return text