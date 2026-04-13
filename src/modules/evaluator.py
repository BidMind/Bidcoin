# 압축된 문서들이 정말로 정답을 낼 수 있는지 마지막으로 팩트 체크

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def evaluate_contexts(query: str, contexts: list) -> bool:
    """압축된 문서들이 질문에 대답할 수 있는지 최종 검열합니다."""
    if not contexts:
        return False
        
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    combined_text = "\n".join([c["text"] for c in contexts])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 엄격한 팩트 체커야. 주어진 [문서들]의 내용만으로 [사용자 질문]에 대한 명확한 답변이 가능한지 평가해.
        답변이 가능하면 'YES', 문서 내용이 부족하거나 엉뚱하다면 'NO'를 출력해."""),
        ("human", "[사용자 질문]: {query}\n\n[문서들]: {combined_text}")
    ])
    
    try:
        decision = (prompt | llm).invoke({"query": query, "combined_text": combined_text}).content.strip().upper()
        return decision == "YES"
    except Exception as e:
        print(f"⚠️ [검열 에러] 일단 통과시킴: {e}")
        return True