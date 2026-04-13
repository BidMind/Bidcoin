# 엔진의 chat_states를 가져오되, 이 파일 안에서는 chat_state라고 부르겠다는 뜻!
from rag1_engine_multiquery import ask_bid_chatbot, chat_states as chat_state

# 1. 폼나게 타이틀 수정!
print("\n🚀 B2G 공공입찰 전문 챗봇이 시작되었습니다! (Multi-Query 하이브리드 검색 탑재 🔥)")
print("-" * 60)

while True:
    try:
        user_input = input("❓ 질문 입력 (종료: q, 근거확인: 팩트): ").strip()
        
        # 1. 종료 로직
        if user_input.lower() in ['q', 'quit', '종료']:
            print("👋 대화를 종료합니다.")
            break
            
        # 2. 팩트체크 로직
        if user_input == "팩트":
            # [수정] s를 제거하고, 기본 세션인 ["default"] 칸을 확인합니다.
            if not chat_state["default"]["last_context"]:
                print("❌ 아직 진행된 대화가 없거나 참조된 근거가 없습니다.")
            else:
                print("\n🔎 [실시간 팩트체크] 방금 답변의 근거입니다.")
                # [수정] 여기도 "default" 칸의 데이터를 가져오도록 변경
                for i, row in enumerate(chat_state["default"]["last_context"]):
                    # 빈 값(None)이 뜨지 않도록 기본값 장착
                    biz_name = row.get('사업명', '알 수 없는 사업')
                    
                    # 하이브리드/멀티쿼리 합산 점수(rrf_score)와 매칭수(query_hits) 가져오기
                    score = row.get('rrf_score', 0) 
                    hits = row.get('query_hits', 1)
                    chunk_text = row.get('청크_텍스트', '')
                    
                    # 점수가 소수점일 경우 깔끔하게 4자리까지만 출력
                    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                        
                    print(f"[{i+1}번] {biz_name} | 합산 점수: {score_str} | 쿼리 매칭수: {hits}")
                    print(f"📄 원문: {chunk_text[:300]}...\n")
            continue
        
        # 3. 빈 입력 방지
        if not user_input: 
            continue

        # 4. 챗봇 엔진 호출
        print("\n🔍 AI가 관련 공고를 분석 중입니다...")
        answer = ask_bid_chatbot(user_input)
        
        # 5. 결과 출력
        print(f"\n🤖 AI 답변:\n{answer}")
        print("\n" + "="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n👋 대화를 강제로 종료합니다.")
        break
    except Exception as e:
        print(f"\n🚨 앗, 일시적인 오류가 발생했습니다! (사유: {e})")
        print("다시 한 번 질문해 주시겠어요?\n" + "="*60 + "\n")