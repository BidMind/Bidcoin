import numpy as np
from kiwipiepy import Kiwi
from src.embedding.vector_store import load_db, load_bm25

# ==========================================
# [전역 설정] 형태소 분석기 초기화
# ==========================================
# 매 검색마다 Kiwi 객체를 새로 만들면 속도가 매우 느려지므로 파일 상단에서 한 번만 생성합니다.
kiwi = Kiwi()

# 불용어(Stopwords) 태그 모음: 의미 검색에 방해되는 조사나 기호들을 걸러냅니다.
STOP_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 
             'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}

def korean_tokenizer(text: str) -> list[str]:
    """
    사용자의 질문을 BM25가 이해할 수 있는 핵심 단어(토큰)들의 리스트로 변환합니다.
    예: "한영대학교 사업의 예산은?" -> ["한영대학교", "사업", "예산"]
    """
    try:
        tokens = kiwi.tokenize(str(text))
        # 불용어가 아니고 길이가 2글자 이상인 단어만 추출합니다.
        return [t.form for t in tokens if t.tag not in STOP_TAGS and len(t.form) > 1]
    except Exception:
        # 에러 발생 시 기본 띄어쓰기 기준으로 대체합니다.
        return str(text).split()

# ==========================================
# [메인 검색기] 하이브리드 검색 함수
# ==========================================
def retrieve_candidates(semantic_query: str, keyword_query: str, k: int = 10):
    """
    [기능] FAISS(의미)와 BM25(키워드) 검색을 결합한 하이브리드 검색을 수행합니다.
    - semantic_query: 의미 기반(벡터) 검색을 위한 긴 텍스트 (예: HyDE로 생성된 가상 문서)
    - keyword_query: 단어 일치(키워드) 검색을 위한 짧은 핵심 질문
    - k: 최종적으로 LLM에 넘길 문서의 개수
    """

    # --------------------------------------------------
    # Step 1. 두 가지 검색 DB 불러오기
    # --------------------------------------------------
    vector_store = load_db()              # FAISS 벡터 인덱스 로드
    bm25_data = load_bm25()               # BM25 인덱스 데이터 로드
    bm25_engine = bm25_data["bm25_obj"]   # 실제 BM25 검색을 수행할 엔진 객체
    all_docs = bm25_data["docs"]          # 인덱스와 매핑된 원본 문서 리스트

    # RRF(순위 결합)를 제대로 하려면 최종 반환 개수(k)보다 더 많은 후보군을 각각 뽑아와야 합니다.
    # 10개를 원한다면 각각 20개씩 뽑아서 합친 뒤 상위 10개를 추립니다.
    candidate_k = max(20, k * 2)

    # --------------------------------------------------
    # Step 2. [Vector] 의미 기반 검색 (FAISS)
    # --------------------------------------------------
    # FAISS를 이용해 문맥이 가장 유사한 문서 candidate_k개를 찾습니다.
    faiss_results = vector_store.similarity_search(semantic_query, k=candidate_k)
    
    # 찾은 문서들의 순위(Rank)를 딕셔너리에 저장합니다. (예: "문서A 내용": 1등, "문서B 내용": 2등)
    faiss_scores = {doc.page_content: rank for rank, doc in enumerate(faiss_results, start=1)}

    # --------------------------------------------------
    # Step 3. [Keyword] 단어 일치 기반 검색 (BM25)
    # --------------------------------------------------
    # 키워드 쿼리를 형태소 분석기로 쪼갭니다.
    tokenized_query = korean_tokenizer(keyword_query)
    
    # BM25 엔진에 넣어 각 문서별 키워드 매칭 점수(배열)를 얻습니다.
    bm25_raw_scores = bm25_engine.get_scores(tokenized_query)
    
    # 점수가 높은 순서대로 인덱스를 나열하고, 상위 candidate_k개만 자릅니다.
    top_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:candidate_k]
    
    bm25_scores = {}
    for rank, idx in enumerate(top_bm25_indices, start=1):
        doc = all_docs[idx]
        # 점수가 0보다 크다는 것은 검색어가 문서에 최소 한 번은 등장했다는 뜻입니다.
        if bm25_raw_scores[idx] > 0: 
            bm25_scores[doc.page_content] = rank # FAISS처럼 순위를 저장합니다.

    # --------------------------------------------------
    # Step 4. 두 검색 결과 병합 (RRF 점수 계산)
    # --------------------------------------------------
    k_rrf = 60 # RRF 수식의 보정 상수 (일반적으로 60을 사용합니다)
    rrf_map = {}
    
    # 나중에 문서 내용을 통해 원본 Document 객체를 찾기 위한 사전입니다.
    content_to_doc = {doc.page_content: doc for doc in all_docs}

    # FAISS 결과와 BM25 결과에서 나온 모든 문서 내용(문자열)의 합집합을 구합니다. (중복 제거)
    all_candidate_contents = set(faiss_scores.keys()) | set(bm25_scores.keys())

    # 합쳐진 모든 문서 후보들을 하나씩 돌면서 최종 점수를 계산합니다.
    for content in all_candidate_contents:
        # FAISS에서 몇 등이었는지 찾습니다. (검색되지 않았다면 꼴등 처리: candidate_k + 1)
        f_rank = faiss_scores.get(content, candidate_k + 1) 
        
        # BM25에서 몇 등이었는지 찾습니다. (검색되지 않았다면 꼴등 처리)
        b_rank = bm25_scores.get(content, candidate_k + 1)
        
        # RRF 공식: (1 / (60 + FAISS순위)) + (1 / (60 + BM25순위))
        # 두 엔진에서 모두 상위권에 있을수록 점수가 기하급수적으로 높아집니다.
        rrf_score = (1 / (k_rrf + f_rank)) + (1 / (k_rrf + b_rank))
        rrf_map[content] = rrf_score

    # --------------------------------------------------
    # Step 5. 최종 결과 정렬 및 반환
    # --------------------------------------------------
    # RRF 점수가 높은 순(내림차순)으로 정렬하고, 최종적으로 요구한 k개만 잘라냅니다.
    sorted_contents = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[:k]
    
    final_candidates = []
    for content, score in sorted_contents:
        doc = content_to_doc[content]
        # 디버깅이나 다음 단계(Re-ranker)에서 점수를 확인할 수 있도록 메타데이터에 RRF 점수를 달아줍니다.
        doc.metadata["rrf_score"] = score 
        final_candidates.append(doc)

    return final_candidates # 상위 k개의 LangChain Document 리스트 반환