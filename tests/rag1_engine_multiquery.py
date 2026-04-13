import os
import time
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

# ---------------------------------------------------------
# 1. 기본 설정 및 API 키 로드
# ---------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DIMENSION = 1536
SYSTEM_PROMPT = """당신은 B2G 공공입찰 전문 컨설팅 어시스턴트입니다.
주어진 RFP 문서 내용만을 기반으로 답변하세요.
문서에 없는 내용은 반드시 "해당 문서에서 확인할 수 없습니다"라고 답하세요.
답변은 구조화된 형식으로 제공하고, 출처 문서명을 항상 명시하세요."""

# ---------------------------------------------------------
# 2. 데이터 로드 및 전역 변수 설정
# ---------------------------------------------------------
print("⏳ 데이터를 불러오는 중입니다...")
df = pd.read_pickle('./bid_master_optimized_v2.pkl')

TOTAL_DOCS = len(df)  
FAISS_CANDIDATE_K = min(20, int(TOTAL_DOCS * 0.2))  
BM25_CANDIDATE_K  = min(20, int(TOTAL_DOCS * 0.2))  
FINAL_CONTEXT_K   = 5 

# ---------------------------------------------------------
# 3. FAISS 및 BM25 인덱스 구축
# ---------------------------------------------------------
index_file = "bid_index.faiss"
index = faiss.read_index(index_file)

kiwi = Kiwi()
STOP_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}

def korean_tokenizer(text: str) -> list[str]:
    try:
        tokens = kiwi.tokenize(str(text))
        return [t.form for t in tokens if t.tag not in STOP_TAGS and len(t.form) > 1]
    except Exception:
        return str(text).split()

print("⏳ BM25 인덱스 구축 중...")
tokenized_corpus = [korean_tokenizer(doc) for doc in df['청크_텍스트']]
bm25 = BM25Okapi(tokenized_corpus)

# ---------------------------------------------------------
# 4. 멀티 쿼리 생성 함수 (LLM 활용)
# ---------------------------------------------------------
def generate_multi_queries(original_query):
    system_msg = "당신은 공공입찰 검색 전문가입니다. 사용자의 질문을 분석하여 검색 정확도를 높일 수 있는 3개의 유사한 검색어로 확장하세요."
    user_msg = f"질문: {original_query}\n\n위 질문과 의미는 같지만 다른 키워드를 포함한 검색어 3개를 줄바꿈으로만 구분해서 나열하세요."
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.7
    )
    
    generated_queries = response.choices[0].message.content.strip().split('\n')
    all_queries = [original_query] + [q.strip() for q in generated_queries if q.strip()]
    return all_queries[:4]

# ---------------------------------------------------------
# 5. 핵심 검색 엔진 (Hybrid Search)
# ---------------------------------------------------------
def search_hybrid(query_text: str, top_k: int = 5) -> pd.DataFrame:
    # 임베딩 생성
    query_response = client.embeddings.create(input=[query_text], model=EMBEDDING_MODEL)
    query_vector = np.array([query_response.data[0].embedding]).astype('float32')
    
    # FAISS 검색
    distances, indices = index.search(query_vector, FAISS_CANDIDATE_K)
    
    # BM25 검색
    query_tokens = korean_tokenizer(query_text)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:BM25_CANDIDATE_K]

    # RRF 결합
    k_rrf = 60
    candidate_indices = set(indices[0]) | set(bm25_top_indices)
    rrf_records = []

    for idx in candidate_indices:
        f_rank = np.where(indices[0] == idx)[0][0] + 1 if idx in indices[0] else FAISS_CANDIDATE_K + 1
        b_rank = np.where(bm25_top_indices == idx)[0][0] + 1 if idx in bm25_top_indices else BM25_CANDIDATE_K + 1
        rrf_score = (1 / (k_rrf + f_rank)) + (1 / (k_rrf + b_rank))
        rrf_records.append({'df_index': idx, 'rrf_score': rrf_score})

    rrf_df = pd.DataFrame(rrf_records).sort_values('rrf_score', ascending=False)
    top_indices = rrf_df.head(top_k)['df_index'].tolist()
    
    result = df.iloc[top_indices].copy()
    result['rrf_score'] = rrf_df.head(top_k)['rrf_score'].values
    return result

# ---------------------------------------------------------
# 6. 멀티 쿼리 통합 검색 (Aggregator)
# ---------------------------------------------------------
def multi_query_search(expanded_queries, top_k=5):
    aggregated = {} # {idx: {'score': 0, 'hits': 0}}
    
    for q in expanded_queries:
        search_result = search_hybrid(q, top_k=3) # 각 쿼리당 상위 3개씩 추출
        for _, row in search_result.iterrows():
            idx = row.name
            if idx not in aggregated:
                aggregated[idx] = {'score': 0, 'hits': 0}
            aggregated[idx]['score'] += row['rrf_score']
            aggregated[idx]['hits'] += 1
            
    sorted_idx = sorted(aggregated.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
    
    final_result = df.loc[[x[0] for x in sorted_idx]].copy()
    final_result['rrf_score'] = [x[1]['score'] for x in sorted_idx]
    final_result['query_hits'] = [x[1]['hits'] for x in sorted_idx]
    return final_result

# ---------------------------------------------------------
# 7. 상태 관리 및 챗봇 메인
# ---------------------------------------------------------
chat_states = {}

def call_llm_with_retry(messages: list) -> str:
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0)
    return response.choices[0].message.content

def ask_bid_chatbot(query: str, session_id: str = "default") -> str:
    state = chat_states.setdefault(session_id, {"memory": [], "last_context": [], "last_queries": []})

    # 1. 쿼리 확장
    expanded_queries = generate_multi_queries(query)
    state["last_queries"] = expanded_queries
    
    # 2. 멀티 쿼리 하이브리드 검색
    try:
        final_docs = multi_query_search(expanded_queries, top_k=FINAL_CONTEXT_K)
    except Exception as e:
        return f"❌ 검색 중 오류가 발생했습니다: {e}"

    state["last_context"] = final_docs.to_dict('records')

    # 3. 컨텍스트 구성
    context = ""
    for _, row in final_docs.iterrows():
        biz_name = row.get('사업명', '알 수 없는 사업')
        bid_no   = row.get('공고번호', '정보 없음')
        hits     = row.get('query_hits', 1)
        context += f"[출처: {biz_name} | 공고번호: {bid_no} | 매칭수: {hits}]\n{row['청크_텍스트']}\n---\n"

    # 4. 답변 생성
    history_text = "".join([f"사용자: {h['q']}\n어시스턴트: {h['a']}\n" for h in state["memory"][-3:]])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 문서]\n{context}\n[히스토리]\n{history_text}\n[질문]\n{query}"}
    ]

    try:
        answer = call_llm_with_retry(messages)
        display_queries = f"💡 AI가 확장한 검색어: {', '.join(expanded_queries[1:])}\n\n"
        full_answer = display_queries + answer
        state["memory"].append({"q": query, "a": answer})
        return full_answer
    except Exception as e:
        return f"❌ 답변 생성 중 오류가 발생했습니다: {e}"