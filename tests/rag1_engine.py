import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. 기본 설정 및 API 키 로드
# ---------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DIMENSION = 1536

# ---------------------------------------------------------
# 2. 데이터 및 FAISS 인덱스 로드 (없으면 자동 생성)
# ---------------------------------------------------------
print("⏳ 데이터 및 FAISS 인덱스를 점검 중입니다...")
df = pd.read_pickle('./bid_master_optimized_v2.pkl')
index_file = "bid_index.faiss"

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    if index.ntotal != len(df):  # 데이터 개수가 달라졌으면 갱신
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

# ---------------------------------------------------------
# 3. 상태 관리 (대화 메모리 및 팩트체크 저장소)
# ---------------------------------------------------------
chat_state = {
    "memory": [],
    "last_context": []
}

SYSTEM_PROMPT = """당신은 B2G 공공입찰 전문 컨설팅 어시스턴트입니다.
주어진 RFP 문서 내용만을 기반으로 답변하세요.
문서에 없는 내용은 반드시 "해당 문서에서 확인할 수 없습니다"라고 답하세요.
답변은 구조화된 형식으로 제공하고, 출처 문서명을 항상 명시하세요."""

# ---------------------------------------------------------
# 4. 검색 및 랭킹 함수
# ---------------------------------------------------------
def search_faiss_pro(query_text, top_k=5):
    query_response = client.embeddings.create(input=[query_text], model=EMBEDDING_MODEL)
    query_vector = np.array([query_response.data[0].embedding]).astype('float32')
    
    distances, indices = index.search(query_vector, top_k * 3) # 넉넉히 검색
    results = df.iloc[indices[0]].copy()
    return results.head(top_k)

def re_rank(query, results_list):
    query_words = [word for word in query.split() if len(word) > 1]
    results_df = pd.DataFrame(results_list)
    
    results_df['re_rank_score'] = results_df['청크_텍스트'].apply(
        lambda text: sum(1 for word in query_words if word.lower() in str(text).lower())
    )
    return results_df.sort_values(by='re_rank_score', ascending=False)

# ---------------------------------------------------------
# 5. 챗봇 응답 생성 메인 함수
# ---------------------------------------------------------
def ask_bid_chatbot(query):
    # 1. 검색 및 리랭킹
    search_results = search_faiss_pro(query, top_k=5)
    sorted_results = re_rank(query, search_results)
    
    # 2. 팩트체크용 원본 데이터 저장
    chat_state["last_context"] = sorted_results.head(3).to_dict('records')
    
    # 3. 컨텍스트 조립
    context = ""
    for _, row in sorted_results.head(3).iterrows():
        biz_name = row.get('사업명', '알 수 없는 사업')
        bid_no = row.get('공고번호', '정보 없음')
        context += f"[출처: {biz_name} | 공고번호: {bid_no}]\n{row['청크_텍스트']}\n---\n"

    # 4. 과거 대화 내역 조립
    history_text = ""
    for hist in chat_state["memory"][-3:]:
        history_text += f"사용자: {hist['q']}\n어시스턴트: {hist['a']}\n"

    # 5. 최종 프롬프트 생성
    final_user_content = f"""[참고 문서]
{context}
[대화 히스토리]
{history_text if history_text else "없음"}
[질문]
{query}
"""

    # 6. LLM 호출
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_user_content}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    chat_state["memory"].append({"q": query, "a": answer}) # 대화 기록 저장
    
    return answer