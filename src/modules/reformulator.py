import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def reformulate_query(query: str, chat_history: list = None) -> dict:
    """
    [기능] 질문을 분석하여 다중 쿼리와 메타데이터 필터 조건을 동시에 추출합니다.
    - 반환 형태: {"queries": ["검색어1"], "filters": {"발주 기관": "한영대학교"}}
    """
    # 1. 대화 기록이 비어있다면 굳이 LLM을 부르지 않고 원본 질문을 그대로 반환 (속도 최적화)
    if chat_history is None:
        chat_history = []
        
    # 2. 모델
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # 3. LLM에게 내릴 명확한 지시사항
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 RAG 검색 엔진의 질문 분석 및 메타데이터 필터 추출 전문가야.
사용자의 질문과 대화 기록을 분석하여, 검색할 '쿼리 리스트'와 문서를 거를 '필터 조건'을 추출해.

[작업 규칙]
1. 비교 질문은 독립된 여러 개의 쿼리로 분할할 것. (예: ["한영대 예산", "한국대 예산"])
2. 사용자의 질문에 특정 '발주 기관'이나 '사업명'이 명시되어 있다면 filters에 추가할 것. 없다면 filters는 비워둘 것.
3. 반드시 아래 JSON 형식으로만 출력할 것. 다른 부연 설명은 절대 금지.

[출력 예시]
{
  "queries": ["한영대학교 학사정보 예산", "한국대학교 차세대 예산"],
  "filters": {
    "발주 기관": "한영대학교, 한국대학교",
    "사업명": "학사정보, 차세대"
  }
}"""),
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
        
        # 마크다운 코드블록(```json)이 붙어있을 경우 제거
        result = re.sub(r"```json\n?|```", "", result).strip()
        
        parsed_data = json.loads(result)
        print(f"🔄 [질문 분석 완료] 쿼리: {parsed_data.get('queries', [])} | 필터: {parsed_data.get('filters', {})}")
        return parsed_data
        
    except Exception as e:
        print(f"⚠️ [분석 에러] 기본 검색으로 폴백 (필터 없음): {e}")
        return {"queries": [query], "filters": {}}        
        
