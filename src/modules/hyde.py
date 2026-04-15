# 검색 명중률을 극한으로 끌어올리기 위해 가짜 입찰 문서를 생성합니다.
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# def generate_hyde_document(query: str) -> str:
#     """질문에 대한 가상의 제안요청서(RFP) 단락을 생성합니다."""
#     llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """너는 공공기관 입찰 문서 작성자야. 
#         사용자의 질문에 대한 가상의 정답을 '실제 입찰 제안요청서(RFP)에 적혀있을 법한 공식적이고 전문적인 문어체'로 1문단(약 200자) 작성해.
#         숫자나 고유명사가 틀려도 상관없으니 구조와 어휘를 입찰 문서처럼 흉내 내는 것에 집중해."""),
#         ("human", "{query}")
#     ])
    
#     try:
#         return (prompt | llm).invoke({"query": query}).content.strip()
#     except Exception as e:
#         print(f"⚠️ [HyDE 에러] 원본 질문으로 대체: {e}")
#         return query


# 수정:  필드조회형 질문은 HyDE 비활성화 ------------------------
# 필드조회형 질문 감지 키워드
FIELD_QUERY_KEYWORDS = [
    # 금액
    "사업금액", "사업 금액", "금액", "얼마", "예산", "비용",

    # 마감일
    "마감일", "마감", "입찰일", "날짜", "기간", "언제까지",
    "입찰 참여 마감일",

    # 시작일
    "시작일", "언제부터", "입찰 참여 시작일",

    # 공고
    "차수", "공고번호", "공고 번호", "재공고",

    # 기관/사업
    "사업명", "발주기관", "발주 기관",
]

def is_field_query(query: str) -> bool:
    """기관명/금액/날짜 등 메타 필드 조회형 질문 감지"""
    return any(kw in query for kw in FIELD_QUERY_KEYWORDS)

def generate_hyde_document(query: str) -> str:
    """질문에 대한 가상의 제안요청서(RFP) 단락을 생성합니다."""

    # 필드조회형 질문은 HyDE가 노이즈 → 원본 쿼리 그대로 반환
    if is_field_query(query):
        print("[HyDE 비활성화] 필드조회형 질문 감지 → 원본 쿼리 사용")
        return query

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
        print(f"[HyDE 에러] 원본 질문으로 대체: {e}")
        return query