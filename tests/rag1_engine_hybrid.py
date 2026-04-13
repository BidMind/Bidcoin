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

# ---------------------------------------------------------
# 2. 데이터 로드 및 전역 변수 설정 (🔥 수정 1: df를 먼저 부름)
# ---------------------------------------------------------
print("⏳ 데이터를 불러오는 중입니다...")
df = pd.read_pickle('./bid_master_optimized_v2.pkl')

TOTAL_DOCS = len(df)  
FAISS_CANDIDATE_K = min(20, int(TOTAL_DOCS * 0.2))  
BM25_CANDIDATE_K  = min(20, int(TOTAL_DOCS * 0.2))  
FINAL_CONTEXT_K   = 5  # LLM에 넘길 최종 문서 수

# ---------------------------------------------------------
# 3. FAISS 인덱스 로드 (의미 검색용)
# ---------------------------------------------------------
print("⏳ FAISS 인덱스를 점검 중입니다...")
index_file = "bid_index.faiss"

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    if index.ntotal != len(df):
        os.remove(index_file)
        index = None
else:
    index = None

if index is None:
    index = faiss.IndexFlatL2(DIMENSION)
    embeddings_matrix = np.vstack(df['embedding'].values).astype('float32')
    index.add(embeddings_matrix)
    faiss.write_index(index, index_file)
    print(f"✅ 새 FAISS 인덱스 생성 완료! (총 {index.ntotal}건)")
else:
    print(f"✅ 기존 FAISS 인덱스 로드 완료! (총 {index.ntotal}건)")

# ---------------------------------------------------------
# 4. 한국어 형태소 분석기 + BM25 인덱스 구축
# ---------------------------------------------------------
print("⏳ 형태소 분석기 및 BM25 인덱스를 구축 중입니다...")

kiwi = Kiwi()

# 제외할 품사 태그 (조사, 구두점, 특수기호 등 불용어)
STOP_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
             'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}

def korean_tokenizer(text: str) -> list[str]:
    """
    kiwipiepy 형태소 분석기를 사용한 한국어 토크나이저.
    """
    try:
        tokens = kiwi.tokenize(str(text))
        return [t.form for t in tokens if t.tag not in STOP_TAGS and len(t.form) > 1]
    except Exception:
        return str(text).split()

print("⏳ BM25 말뭉치 토크나이징 중... (문서 수에 따라 약간의 시간이 소요됩니다)")
tokenized_corpus = [korean_tokenizer(doc) for doc in df['청크_텍스트']]
bm25 = BM25Okapi(tokenized_corpus)

def get_bm25_all_ranks(query_tokens: list[str]) -> pd.Series:
    scores = bm25.get_scores(query_tokens)
    # 🔥 수정 3: 원본 데이터(df)의 인덱스와 정확히 매칭
    return pd.Series(scores, index=df.index).rank(ascending=False, method='first')

print("✅ BM25 인덱스 준비 완료!")

# ---------------------------------------------------------
# 5. 임베딩 API 호출 (재시도 로직 포함)
# ---------------------------------------------------------
def get_embedding_with_retry(text: str, retries: int = 3) -> np.ndarray:
    for i in range(retries):
        try:
            response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
            return np.array([response.data[0].embedding]).astype('float32')
        except Exception as e:
            if i == retries - 1:
                raise RuntimeError(f"임베딩 API 호출 실패 ({retries}회 시도): {e}")
            wait = 2 ** i
            print(f"⚠️ 임베딩 API 오류, {wait}초 후 재시도... ({e})")
            time.sleep(wait)

# ---------------------------------------------------------
# 6. 상태 관리 (🔥 수정 2: 기존 run_chatbot.py 호환성 유지)
# ---------------------------------------------------------
# run_chatbot.py가 바로 참조할 수 있도록 기본 chat_state 유지
chat_state = {"memory": [], "last_context": []}

# 세션 기반 멀티유저 지원을 위한 딕셔너리 (기본값으로 위 chat_state 연결)
chat_states: dict = {"default": chat_state} 

SYSTEM_PROMPT = """당신은 B2G 공공입찰 전문 컨설팅 어시스턴트입니다.
주어진 RFP 문서 내용만을 기반으로 답변하세요.
문서에 없는 내용은 반드시 "해당 문서에서 확인할 수 없습니다"라고 답하세요.
답변은 구조화된 형식으로 제공하고, 출처 문서명을 항상 명시하세요."""

# ---------------------------------------------------------
# 7. 하이브리드 검색 (FAISS + BM25 + RRF)
# ---------------------------------------------------------
def search_hybrid(query_text: str, top_k: int = FINAL_CONTEXT_K) -> pd.DataFrame:
    # ── FAISS 검색 ──
    query_vector = get_embedding_with_retry(query_text)
    distances, indices = index.search(query_vector, FAISS_CANDIDATE_K)

    faiss_results = df.iloc[indices[0]].copy()
    faiss_results['faiss_rank'] = range(1, len(faiss_results) + 1)
    faiss_results['faiss_score'] = 1 / (1 + distances[0])

    # ── BM25 검색 ──
    query_tokens = korean_tokenizer(query_text)
    bm25_all_ranks = get_bm25_all_ranks(query_tokens)

    bm25_top_indices = bm25_all_ranks.nsmallest(BM25_CANDIDATE_K).index
    bm25_top_df = df.iloc[bm25_top_indices].copy()
    bm25_top_df['bm25_rank'] = [bm25_all_ranks[i] for i in bm25_top_indices]

    # ── RRF 결합 ──
    k_rrf = 60 
    candidate_indices = set(faiss_results.index) | set(bm25_top_df.index)
    rrf_records = []

    for idx in candidate_indices:
        if idx in faiss_results.index:
            f_rank = faiss_results.loc[idx, 'faiss_rank']
        else:
            f_rank = FAISS_CANDIDATE_K + 1

        b_rank = bm25_all_ranks[idx]

        rrf_score = (1 / (k_rrf + f_rank)) + (1 / (k_rrf + b_rank))
        rrf_records.append({'df_index': idx, 'rrf_score': rrf_score})

    rrf_df = pd.DataFrame(rrf_records).sort_values('rrf_score', ascending=False)
    top_indices = rrf_df.head(top_k)['df_index'].tolist()

    result = df.loc[top_indices].copy()
    result['rrf_score'] = rrf_df.head(top_k)['rrf_score'].values
    return result

# ---------------------------------------------------------
# 8. LLM 호출 (재시도 로직 포함)
# ---------------------------------------------------------
def call_llm_with_retry(messages: list, retries: int = 3) -> str:
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            if i == retries - 1:
                raise RuntimeError(f"LLM API 호출 실패 ({retries}회 시도): {e}")
            wait = 2 ** i
            print(f"⚠️ LLM API 오류, {wait}초 후 재시도... ({e})")
            time.sleep(wait)

# ---------------------------------------------------------
# 9. 챗봇 응답 생성 메인 함수
# ---------------------------------------------------------
def ask_bid_chatbot(query: str, session_id: str = "default") -> str:
    state = chat_states.setdefault(session_id, {"memory": [], "last_context": []})

    try:
        sorted_results = search_hybrid(query, top_k=FINAL_CONTEXT_K)
    except Exception as e:
        return f"❌ 검색 중 오류가 발생했습니다: {e}"

    state["last_context"] = sorted_results.to_dict('records')

    context = ""
    for _, row in sorted_results.iterrows():
        biz_name = row.get('사업명', '알 수 없는 사업')
        bid_no   = row.get('공고번호', '정보 없음')
        context += f"[출처: {biz_name} | 공고번호: {bid_no}]\n{row['청크_텍스트']}\n---\n"

    history_text = ""
    for hist in state["memory"][-3:]:
        history_text += f"사용자: {hist['q']}\n어시스턴트: {hist['a']}\n"

    final_user_content = (
        f"[참고 문서]\n{context}\n"
        f"[대화 히스토리]\n{history_text if history_text else '없음'}\n"
        f"[질문]\n{query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": final_user_content}
    ]

    try:
        answer = call_llm_with_retry(messages)
    except Exception as e:
        return f"❌ 답변 생성 중 오류가 발생했습니다: {e}"

    state["memory"].append({"q": query, "a": answer})

    return answer