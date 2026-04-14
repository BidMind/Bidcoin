# 과거 대화 기록을 바탕으로 생략된 주어나 맥락을 채워 넣습니다.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def reformulate_query(query: str, chat_history: list = None) -> str:
    """
    [기능] 이전 대화 기록을 참고하여 현재 질문을 완벽한 독립 검색어로 재구성합니다.
    """
    # 1. 대화 기록이 비어있다면 굳이 LLM을 부르지 않고 원본 질문을 그대로 반환 (속도 최적화)
    if not chat_history:
        return query
        
    # 2. 모델
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # 3. LLM에게 내릴 명확한 지시사항
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 RAG 검색 엔진을 위한 질문 재구성 전문가야.
        사용자의 [현재 질문]이 대명사로 되어 있거나 맥락이 생략되어 있다면,
        [대화 기록]을 참고해서 검색 엔진이 정확히 문서를 찾을 수 있도록 구체적이고 독립적인 단일 문장으로 재구성해.
        질문에 대답하지 말고 오직 '재구성된 질문'만 출력해."""),
        ("human", "대화 기록:\n{history}\n\n현재 질문: {question}")
    ])
    
    # 4. 딕셔너리 형태의 대화 기록을 LLM이 읽기 편한 문자열로 변환
    history_str = "\n".join([
        f"{'User' if msg.get('role') == 'user' else 'Bot'}: {msg.get('content', '')}"
        for msg in chat_history
    ])
    # 5. 실행 및 무중단 에러 방어
    try:
        chain = prompt | llm
        rewritten_query = chain.invoke({"history": history_str, "question": query}).content.strip()
        print(f"🔄 [질문 재구성 완료] {query} ➡️ {rewritten_query}")
        return rewritten_query
    except Exception as e:
        # 혹시 OpenAI 서버가 응답하지 않더라도 시스템이 죽지 않고 원본 질문으로 검색하도록 처리
        print(f"⚠️ [재구성 모듈 에러] 원본 질문으로 검색 진행: {e}")
        return query