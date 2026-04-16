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
# --- [신규 추가] 필터 검사 함수 ---
def pass_metadata_filter(metadata: dict, filters: dict) -> bool:
    """문서의 메타데이터가 필터 조건(발주 기관, 사업명 등)을 포함하는지 검사합니다."""
    if not filters:
        return True # 필터가 없으면 무조건 통과
        
    for filter_key, filter_val in filters.items():
        if filter_key in metadata:
            doc_val = str(metadata[filter_key]).replace(" ", "")
            target_vals = [v.strip().replace(" ", "") for v in filter_val.split(",")]
            
            # 여러 타겟(예: 한영대, 한국대) 중 하나라도 문서 메타데이터에 포함되어 있으면 통과
            if any(target in doc_val or doc_val in target for target in target_vals):
                continue
            return False # 하나라도 매칭 실패 시 즉시 탈락
    return True

# [수정]
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
    filters = filters or {}               # 파라미터에 filters: dict 추가
    vector_store = load_db()              # FAISS 벡터 인덱스 로드
    bm25_data = load_bm25()               # BM25 인덱스 데이터 로드
    bm25_engine = bm25_data["bm25_obj"]   # 실제 BM25 검색을 수행할 엔진 객체
    all_docs = bm25_data["docs"]          # 인덱스와 매핑된 원본 문서 리스트

    # 메타데이터에서 많이 걸러질 것을 대비해 FAISS는 평소보다 3배 더 넉넉히 가져옵니다.
    candidate_k = max(20, k * 2)
    fetch_k = candidate_k * 3

    # --------------------------------------------------
    # Step 2. [Vector] FAISS 검색 및 필터링
    # --------------------------------------------------
    raw_faiss_results = vector_store.similarity_search(semantic_query, k=fetch_k)
    faiss_scores = {}
    rank = 1
    for doc in raw_faiss_results:
        if pass_metadata_filter(doc.metadata, filters):
            faiss_scores[doc.page_content] = rank
            rank += 1
            if rank > candidate_k: break # 목표 개수 채우면 중단

    # --------------------------------------------------
    # Step 3. [Keyword] BM25 검색 및 필터링
    # --------------------------------------------------
    # 키워드 쿼리를 형태소 분석기로 쪼갭니다.
    tokenized_query = korean_tokenizer(keyword_query)
    
    # BM25 엔진에 넣어 각 문서별 키워드 매칭 점수(배열)를 얻습니다.
    bm25_raw_scores = bm25_engine.get_scores(tokenized_query)
    
    # 점수가 높은 순서대로 인덱스를 나열하고, 상위 candidate_k개만 자릅니다.
    top_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:candidate_k]
    
    bm25_scores = {}
    rank = 1
    for idx in top_bm25_indices:
        doc = all_docs[idx]
        if bm25_raw_scores[idx] > 0 and pass_metadata_filter(doc.metadata, filters):
            bm25_scores[doc.page_content] = rank
            rank += 1
            if rank > candidate_k: break

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