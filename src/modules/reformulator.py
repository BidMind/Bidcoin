# 과거 대화 기록을 바탕으로 생략된 주어나 맥락을 채워 넣습니다.

import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def reformulate_query(query: str, chat_history: list = None) -> str:
    """
    [기능] 질문을 분석하여 다중 쿼리(Multi-Query) 리스트를 반환합니다.
    - 비교 질문: 각 대상에 대한 개별 쿼리로 분할 (예: ["A 사업 예산", "B 사업 예산"])
    - 일반 질문: 단일 쿼리로 재구성하여 리스트에 담아 반환
    """
    # 1. 대화 기록이 비어있다면 굳이 LLM을 부르지 않고 원본 질문을 그대로 반환 (속도 최적화)
    if chat_history is None:
        chat_history = []
        
    # 2. 모델
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # 3. LLM에게 내릴 명확한 지시사항
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 RAG 검색 엔진을 위한 질문 분석 및 다중 쿼리 생성 전문가야.
사용자의 [현재 질문]과 [대화 기록]을 분석하여, 검색 엔진이 문서를 가장 잘 찾을 수 있는 '검색어 리스트'를 만들어야 해.

[작업 규칙]
1. 대명사나 생략된 맥락이 있다면 대화 기록을 참고해 구체적인 단어로 채워 넣어.
2. 질문이 2개 이상의 대상(예: 사업, 기관 등)을 '비교'하거나 '차이'를 묻는 내용이라면, 각 대상을 독립적으로 검색할 수 있도록 질문을 쪼개서 여러 개의 쿼리로 만들어.
   (예: "한영대학교와 한국대학교 사업 예산 비교해줘" -> "한영대학교 사업 예산", "한국대학교 사업 예산")
3. 일반적인 단일 질문이라면 1개의 완벽한 문장으로만 만들어.
4. 생성된 각 쿼리는 반드시 줄바꿈(Enter)으로만 구분해서 출력해. 번호 매기기나 부연 설명은 절대 하지 마."""),
        ("human", "[대화 기록]:\n{history}\n\n[현재 질문]: {question}")
    ])
    
    # 4. 딕셔너리 형태의 대화 기록을 LLM이 읽기 편한 문자열로 변환
    history_str = "\n".join([
        f"{'User' if msg.get('role') == 'user' else 'Bot'}: {msg.get('content', '')}"
        for msg in chat_history
    ])
    
    # 5. 실행 및 무중단 에러 방어
    try:
        chain = prompt | llm
        result = chain.invoke({"history": history_str, "question": query}).content.strip()
        
        # 줄바꿈으로 분리 후 빈 문자열 제거
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # LLM이 지시를 무시하고 "1. A사업" 처럼 번호를 매겼을 경우를 대비한 정제 로직
        cleaned_queries = [re.sub(r"^\d+[\.\)]\s*", "", q) for q in queries]
        
        print(f"🔄 [질문 재구성 및 다중 쿼리 분할 완료] \n   원본: {query} \n   분할: {cleaned_queries}")
        return cleaned_queries
        
    except Exception as e:
        print(f"⚠️ [재구성 모듈 에러] 원본 질문 단일 쿼리로 진행: {e}")
        return [query] # 에러 시 리스트 형태로 원본 반환
