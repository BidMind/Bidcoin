import os
import pandas as pd
import json

# 1. 방금 만든 모듈들 불러오기
import config
from src.embedding.vector_store import build_and_save_db
from rag_api import get_rag_context

print("🚀 [Phase 1] 가짜 입찰 데이터 생성 중...")

# 2. 테스트용 가짜 데이터(Dummy) 생성
dummy_data = [
    {
        "content": "본 사업은 한국대학교 차세대 학사시스템 구축 사업입니다. 입찰 참가 자격은 소프트웨어 진흥법 제58조에 의한 소프트웨어 사업자로 신고를 필한 업체여야 합니다. 총 예산은 15억 원(VAT 포함)이며, 제안서 제출 마감은 2026년 5월 10일입니다.",
        "source": "한국대학교_학사시스템구축.pdf"
    },
    {
        "content": "미래병원 의료정보시스템(HIS) 클라우드 전환 사업 공고. 필수 제출 서류는 다음과 같습니다: 1. 제안서 원본 1부 및 사본 5부, 2. 사업자등록증 사본 1부, 3. 최근 3년간 500병상 이상 종합병원 실적증명서. 지체상금율은 지연일수 당 1.5/1000 입니다.",
        "source": "미래병원_HIS전환.hwp"
    },
    {
        "content": "제일은행 대국민 AI 챗봇 서비스 고도화 제안요청서(RFP). 입찰 제한 요건: 최근 3년 이내 금융권 AI 챗봇 구축 납품 실적이 없는 자는 본 입찰에 참여할 수 없습니다. 핵심 과업은 LLM(거대언어모델)을 활용한 금융 상품 추천 기능 개발입니다.",
        "source": "제일은행_AI챗봇.pdf"
    }
]

# 3. Dummy 데이터를 processed_data.csv로 강제 저장
df_dummy = pd.DataFrame(dummy_data)
df_dummy.to_csv(config.CSV_PATH, index=False, encoding='utf-8-sig')
print(f"✅ 가짜 데이터 CSV 저장 완료: {config.CSV_PATH}\n")

# 4. DB 구축 모듈 실행
print("🚀 [Phase 2] 임베딩 및 FAISS 벡터 DB 구축 중...")
build_and_save_db()  # src.embedding.vector_store 내부의 함수 실행
print("\n")

# 5. 최종 RAG API 모듈 테스트
test_query = "미래병원 클라우드 전환 사업에 참여하려고 하는데, 어떤 서류를 내야 해?"
print(f"🚀 [Phase 3] 질문 테스트: '{test_query}'")

# RAG 결과 뽑아오기
final_result = get_rag_context(query=test_query)

# 6. LLM 팀원에게 넘어갈 예쁘게 포장된 JSON 결과 출력
print("\n" + "="*60)
print("🎯 [최종 출력 JSON 결과 확인]")
print("="*60)
print(json.dumps(final_result, indent=2, ensure_ascii=False))