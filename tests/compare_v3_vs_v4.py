import sys
import os
from pathlib import Path

# [경로 인식 에러 해결] 현재 파일(tests)의 부모 폴더(프로젝트 루트)를 파이썬 경로에 강제 추가
# 주의: 이 코드는 반드시 from src... 또는 import rag_api... 보다 위에 있어야 합니다!
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------
# 기존 import 코드 시작
import time
import pandas as pd
from src.generation.schemas import RetrievalResult

# V3와 V4 메인 함수 임포트
from rag_api_v3 import get_rag_context as run_v3
from rag_api_v4 import get_rag_context as run_v4

def run_benchmark():
    print("🚀 V3(HyDE 포함) vs V4(HyDE 제거) 성능 벤치마크 시작...\n")
    
    # 테스트할 다양한 난이도의 질문 셋 (일반, 비교, 필터링 등)
    test_queries = [
        "한영대학교 학사정보시스템 고도화 사업의 총 예산은 얼마야?", # 일반 팩트
        "한국대학교 사업과 한영대학교 사업의 입찰 마감일을 비교해 줘", # 다중 쿼리(비교)
        "입찰 참가 자격에 소프트웨어사업자 신고가 필요한 사업은?", # 의미 검색 위주
        "없는 기관인 우주대학교의 AI 시스템 예산 알려줘" # 예외 처리(빈 결과) 테스트
    ]

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] 질의: {query}")

        # ----------------------------------------
        # 1. V3 (HyDE 포함) 구동 및 측정
        # ----------------------------------------
        start_v3 = time.time()
        res_v3 = run_v3(query)
        end_v3 = time.time()
        
        time_v3 = round(end_v3 - start_v3, 2)
        docs_v3_count = len(res_v3.get("contexts", []))
        
        # V3가 가져온 핵심 파일명 추출
        sources_v3 = ", ".join([c["source_file"] for c in res_v3.get("contexts", [])])

        # ----------------------------------------
        # 2. V4 (HyDE 제거 경량화) 구동 및 측정
        # ----------------------------------------
        start_v4 = time.time()
        res_v4 = run_v4(query)
        end_v4 = time.time()
        
        time_v4 = round(end_v4 - start_v4, 2)
        docs_v4_count = len(res_v4.get("contexts", []))
        
        # V4가 가져온 핵심 파일명 추출
        sources_v4 = ", ".join([c["source_file"] for c in res_v4.get("contexts", [])])

        # ----------------------------------------
        # 3. 결과 기록
        # ----------------------------------------
        results.append({
            "테스트 질의": query,
            "V3 속도 (초)": time_v3,
            "V4 속도 (초)": time_v4,
            "단축된 시간": round(time_v3 - time_v4, 2),
            "V3 검색 문서": f"{docs_v3_count}개 ({sources_v3})",
            "V4 검색 문서": f"{docs_v4_count}개 ({sources_v4})"
        })
        print(f"   ▶ V3: {time_v3}초 | V4: {time_v4}초 (단축: {round(time_v3 - time_v4, 2)}초)\n")

    # 결과 통계 출력 및 저장
    df = pd.DataFrame(results)
    print("=" * 80)
    print(" 📊 최종 벤치마크 결과 요약")
    print("=" * 80)
    # 터미널에 표 형태로 예쁘게 출력
    print(df.to_markdown(index=False))
    
    # 결과를 outputs 폴더에 CSV로 저장하여 나중에 분석할 수 있도록 함
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # outputs 폴더가 없으면 마법처럼 생성!
        
    output_path = f"{output_dir}/v3_vs_v4_benchmark_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 상세 결과가 {output_path} 에 저장되었습니다.")

if __name__ == "__main__":
    run_benchmark()