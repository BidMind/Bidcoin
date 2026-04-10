import pandas as pd
from tqdm import tqdm
from kiwipiepy import Kiwi
from chunk1_processor import clean_and_chunk 

print("🚀 [Step 2] 정보 보존형 정제 ➡️ 청킹 ➡️ 형태소 분석 파이프라인 시작...\n")

# 1. 원본 데이터 불러오기
file_path = '/home/bidcoin/bid_master_cleaned.csv'  # 👈 나중에 파일 수정
df = pd.read_csv(file_path)
print(f"✅ 원본 데이터 로드 완료: 총 {len(df)}건\n")

# ------------------------------------------------------------
# [NEW] Kiwi 형태소 분석기 세팅
# ------------------------------------------------------------
kiwi = Kiwi()
STOP_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
             'JX', 'JC', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}

def korean_tokenizer(text):
    # 🚨 텍스트가 비어있으면 에러가 나지 않게 빈 리스트 반환
    if pd.isna(text) or str(text).strip() == "":
        return []
    try:
        tokens = kiwi.tokenize(str(text))
        return [t.form for t in tokens if t.tag not in STOP_TAGS and len(t.form) > 1]
    except Exception:
        return str(text).split()

# ------------------------------------------------------------
# 2. DataFrame의 각 행을 모듈화된 함수로 전달하는 래퍼 함수
def apply_chunking(row):
    return clean_and_chunk(
        raw_text=row.get('텍스트'),
        notice_id=row.get('공고 번호'),
        agency=row.get('발주 기관'),
        project_name=row.get('사업명'),
        amt_val=row.get('사업 금액'),
        start_date=row.get('입찰 참여 시작일')
    )

# 3. 파이프라인 실행
print("✂️ 정제, 청킹, 메타데이터 주입을 진행합니다...")
tqdm.pandas()
df['청크_리스트'] = df.progress_apply(apply_chunking, axis=1)

# 4. 행 분리 (Explode) 및 안전장치
chunked_df = df.explode('청크_리스트').reset_index(drop=True)
chunked_df = chunked_df.rename(columns={'청크_리스트': '청크_텍스트'})

# 🚨 [안전장치] 간혹 빈 텍스트가 섞여서 에러가 나는 것을 방지
chunked_df = chunked_df.dropna(subset=['청크_텍스트']) 

# 5. 불필요 컬럼 제거 및 길이 계산
chunked_df = chunked_df.drop(columns=['텍스트', '텍스트길이'], errors='ignore')
chunked_df['청크_길이'] = chunked_df['청크_텍스트'].apply(lambda x: len(str(x)))

# ------------------------------------------------------------
# 6. [NEW] BM25용 형태소 토큰 미리 만들어두기!
# ------------------------------------------------------------
print("🧠 BM25 검색을 위한 형태소 분석(토크나이징)을 미리 진행합니다...")
chunked_df['BM25_토큰'] = chunked_df['청크_텍스트'].progress_apply(korean_tokenizer)

# 7. 최종 데이터 저장 (CSV + PKL)
csv_path = './bid_master_chunked_v2.csv'
pkl_path = './bid_master_chunked_v2.pkl'

chunked_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
chunked_df.to_pickle(pkl_path) # 파이썬 전용 데이터 파일 (리스트 보존)

print("-" * 60)
print(f"✨ 작업 완료! 원본 {len(df)}건 -> 총 {len(chunked_df)}개의 청크 및 토큰 생성.")
print(f"💾 CSV 저장 완료: '{csv_path}'")
print(f"💾 PKL 저장 완료: '{pkl_path}'")

print("\n[미리보기]")
print(chunked_df[['사업명', '청크_길이', 'BM25_토큰']].head(3))