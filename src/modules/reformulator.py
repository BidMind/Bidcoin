import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# def reformulate_query(query: str, chat_history: list = None) -> dict:
#     """
#     [기능] 질문을 분석하여 다중 쿼리와 메타데이터 필터 조건을 동시에 추출합니다.
#     - 반환 형태: {"queries": ["검색어1"], "filters": {"발주 기관": "한영대학교"}}
#     """
#     # 1. 대화 기록이 비어있다면 굳이 LLM을 부르지 않고 원본 질문을 그대로 반환 (속도 최적화)
#     if chat_history is None:
#         chat_history = []
        
#     # 2. 모델
#     llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
#     # 3. LLM에게 내릴 명확한 지시사항
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """너는 RAG 검색 엔진의 질문 분석 및 메타데이터 필터 추출 전문가야.
#         사용자의 질문과 대화 기록을 분석하여, 검색할 '쿼리 리스트'와 문서를 거를 '필터 조건'을 추출해.

#         [작업 규칙]
#         1. 비교 질문은 독립된 여러 개의 쿼리로 분할할 것. (예: ["한영대 예산", "한국대 예산"])
#         2. 사용자의 질문에 특정 '발주 기관'이나 '사업명'이 명시되어 있다면 filters에 추가할 것. 없다면 filters는 비워둘 것.
#         3. 반드시 아래 JSON 형식으로만 출력할 것. 다른 부연 설명은 절대 금지.

#         [출력 예시]
#         {{
#         "queries": ["클라우드 전환 및 정보시스템 이전 관련 구축 또는 고도화 공고"],
#         "filters": {{}}
#         }}

#         [출력 예시 2]
#         {{
#         "queries": ["한영대학교 학사정보시스템 구축 예산", "한국대학교 차세대 시스템 구축 예산"],
#         "filters": {{
#             "발주 기관": "한영대학교, 한국대학교",
#             "사업명": "학사정보시스템, 차세대 시스템"
#         }}
#         }}
#         """),
#             ("human", "[대화 기록]:\n{history}\n\n[현재 질문]: {question}")
#         ])
    
#     # 4. 딕셔너리 형태의 대화 기록을 LLM이 읽기 편한 문자열로 변환
#     history_str = "\n".join([
#         f"{'User' if msg.get('role') == 'user' else 'Bot'}: {msg.get('content', '')}"
#         for msg in chat_history
#     ])

#     # 5. 실행 및 무중단 에러 방어
#     try:
#         chain = prompt | llm
#         result = chain.invoke({"history": history_str, "question": query}).content.strip()
        
#         # 마크다운 코드블록(```json)이 붙어있을 경우 제거
#         result = re.sub(r"```json\n?|```", "", result).strip()
        
#         parsed_data = json.loads(result)
#         print(f"🔄 [질문 분석 완료] 쿼리: {parsed_data.get('queries', [])} | 필터: {parsed_data.get('filters', {})}")
#         return parsed_data
        
#     except Exception as e:
#         print(f"⚠️ [분석 에러] 기본 검색으로 폴백 (필터 없음): {e}")
#         return {"queries": [query], "filters": {}}        


# # ========== 프롬프트 수정버전 ==========
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
        ("system", """
         너는 RAG 검색 엔진의 질문 분석, 문맥 복원, 검색 질의 정규화, 메타데이터 필터 추출 전문가야.
         사용자의 현재 질문과 대화 기록을 분석하여 다음 작업을 수행하라.

        [작업 목표]
        1. 대화 기록을 참고해 생략된 대상이나 대명사를 복원한다.
        2. 추천/판단/대화형 메타표현이 포함된 질문은 실제 공고문/제안요청서/사업설명에서 검색 가능한 표현으로 정규화한다.
        3. 검색에 유용한 메타데이터 필터(발주기관, 사업명, 금액, 일정 등)가 있으면 추출한다.
        4. 비교 질문은 독립된 여러 개의 검색 질의로 분리한다.

        [정규화 원칙]
        - '추천해줘', '알려줘', '찾아줘', '적합한', '회사에 맞는' 같은 메타표현은 제거한다.
        - 질문의 핵심 의미는 유지하되, 검색 대상이 되는 내용어(기술, 사업, 기관, 조건, 일정 등)만 남긴다.
        - 질문에 없는 기술, 제품명, 방법론, 약어를 새로 추가하지 않는다.
        - 업체 소개문이나 평가문이 아니라, 사업 범위/요구사항/사업유형 중심의 공고형 표현으로 바꾼다.
        - 완전한 설명문이 아니라 짧고 검색 가능한 질의 형태로 만든다.

        [질문 유형별 처리]
        - 금액/조건 → 금액, 자격, 조건이 드러나는 공고 질의
        - 기술/역량 → 해당 기술 또는 업무 범위가 드러나는 사업/요구사항 질의
        - 기관/분야 → 기관명, 분야명, 사업명 중심 공고 질의
        - 마감/일정 → 일정, 기간, 마감일 중심 공고 질의
        - 사업유형 → 구축, 운영, 유지보수, 고도화, 이전, 전환, ISP, 컨설팅 등 사업 성격 중심 질의

        [필터 추출 규칙]
        - 질문에 특정 발주기관, 사업명, 금액, 일정 등이 명시되어 있으면 filters에 넣는다.
        - 명시되지 않으면 해당 filters는 비워둔다.

        [출력 규칙]
        - 반드시 아래 JSON 형식으로만 출력한다.
        - 다른 설명은 절대 출력하지 않는다.

        [출력 예시]
        {{
        "queries": ["클라우드 전환 및 정보시스템 이전 관련 구축 또는 고도화 공고"],
        "filters": {{}}
        }}

        [출력 예시 2]
        {{
        "queries": ["한영대학교 학사정보시스템 구축 예산", "한국대학교 차세대 시스템 구축 예산"],
        "filters": {{
            "발주 기관": "한영대학교, 한국대학교",
            "사업명": "학사정보시스템, 차세대 시스템"
        }}
        }}
        """),
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
        
