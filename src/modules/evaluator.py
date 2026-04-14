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
        ("system", """당신은 검색된 문서들이 질문에 답할 수 있는지 판단하는 검열관입니다.
완벽하고 직접적인 정답이 없더라도, 주어진 문서들을 조합하여 질문에 대한 '부분적인 답변'이나 '관련 배경지식'을 제공할 수 있다면 승인(True) 하세요.
정보가 부족하여 명백히 '환각(Hallucination)'을 일으킬 위험이 있을 때만 거절(False)하세요."""),
        ("human", "[사용자 질문]: {query}\n\n[문서들]: {combined_text}")
    ])
    
    try:
        decision = (prompt | llm).invoke({"query": query, "combined_text": combined_text}).content.strip().upper()
        return decision == "YES"
    except Exception as e:
        print(f"⚠️ [검열 에러] 일단 통과시킴: {e}")
        return True