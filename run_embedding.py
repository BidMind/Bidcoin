import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

print("🚀 OpenAI 텍스트 임베딩 파이프라인 시작...\n")

# 1. 환경변수 및 OpenAI 클라이언트 설정
load_dotenv()
MY_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=MY_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------
# 2. 데이터 로드 (🔥 핵심 수정: CSV 대신 PKL 읽기!)
# ---------------------------------------------------------
file_path = './bid_master_chunked_v2.pkl'
df = pd.read_pickle(file_path)

df.rename(columns={
    '공고 번호': '공고번호',
    '텍스트': '청크_텍스트' 
}, inplace=True, errors='ignore')

print(f"✅ 원본 데이터(PKL) 로드 완료: 총 {len(df)}건")

# ---------------------------------------------------------
# 3. 임베딩 함수 (🔥 보너스 수정: API 에러 방지용 재시도 로직)
# ---------------------------------------------------------
def get_embedding(text, retries=3):
    if not text or pd.isna(text):
        return None
    text = str(text).replace("\n", " ")
    
    for i in range(retries):
        try:
            return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
        except Exception as e:
            if i == retries - 1: # 마지막 시도까지 실패하면
                print(f"\n❌ 임베딩 완전 실패: {e}")
                return None
            wait = 2 ** i # 1초, 2초, 4초... 점점 길게 대기
            time.sleep(wait)

# 4. 임베딩 실행
print(f"🧠 임베딩 변환을 시작합니다. (모델: {EMBEDDING_MODEL})")
tqdm.pandas()
df['embedding'] = df['청크_텍스트'].progress_apply(get_embedding)

# 🚨 [안전장치] 혹시라도 임베딩이 실패해서 None이 들어간 행이 있다면 제거
df = df.dropna(subset=['embedding'])

# 5. 결과 저장 (Pickle)
output_path = './bid_master_optimized_v2.pkl'
df.to_pickle(output_path)

print("-" * 60)
print(f"✨ 임베딩 작업이 성공적으로 완료되었습니다!")
print(f"💾 데이터베이스 저장용 파일: '{output_path}'")