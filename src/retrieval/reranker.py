# cross-encoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def rerank_documents(query: str, candidates: list, top_n: int = 3):
    """
    1차 검색된 문서(candidates)를 질문(query)과 1:1로 비교하여
    Cross-Encoder 모델로 정확도 점수를 매기고 상위 top_n개를 반환합니다.
    """
    if not candidates:
        print("❌ 재정렬할 후보 문서가 없습니다.")
        return []

    print(f"\n[Step 3] BGE-Reranker v2-m3 모델로 정밀 재정렬을 시작합니다...")
    
    # 1. 검증된 BGE Reranker 모델 설정
    model_name = "BAAI/bge-reranker-v2-m3"
    
    # 2. 로컬 GPU 사용 가능 여부 자동 감지
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ⚙️ 연산 디바이스: {device} (GPU 가속 시 매우 빠름)")
    
    # 3. 모델 및 토크나이저 로드
    # (실무에서는 API 서버를 띄워 메모리에 상주시킵니다. 여기서는 호출 시마다 로드합니다.)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval() # 추론 모드로 설정

    # 4. 각 문서에 대해 질문과 연관성(Score) 계산
    scored_docs = []
    print(f" {len(candidates)}개의 문서를 질문과 대조하여 채점 중...")
    
    for doc in candidates:
        # 질문과 문서를 동시에(Cross) 입력으로 넣습니다.
        inputs = tokenizer(
            query, 
            doc.page_content, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            # 모델 출력값에서 점수(logit) 추출
            score = model(**inputs).logits[0][0].item()
            
        scored_docs.append((score, doc))

    # 5. 점수가 가장 높은 순으로 내림차순 정렬
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 6. 상위 top_n개만 잘라내기
    best_docs = scored_docs[:top_n]
    
    print(f" 재정렬 완료! 최고 점수: {best_docs[0][0]:.4f}")
    
    # (점수, 문서객체) 형태의 리스트를 반환
    return best_docs

# ==========================================
# 파이프라인 통합 테스트
# ==========================================
if __name__ == "__main__":
    # search_db 모듈에서 검색 함수를 불러옵니다.
    from search_db import search_from_saved_db
    
    test_query = "입찰 참여를 위한 필수 자격 요건은 무엇인가요?"
    
    # [1단계] FAISS에서 1차 후보군 15개 추출 (빠름)
    faiss_candidates = search_from_saved_db(test_query, k=15)
    
    if faiss_candidates:
        # [2단계] Cross-Encoder로 상위 3개 정밀 추출 (정확함)
        final_top3 = rerank_documents(test_query, faiss_candidates, top_n=3)
        
        # 최종 결과 확인
        print("\n" + "="*60)
        print("🎯 [최종 선별된 핵심 문서 Top 3]")
        print("="*60)
        for i, (score, doc) in enumerate(final_top3):
            print(f"[{i+1}위 | 유사도 점수: {score:.2f}] 출처: {doc.metadata.get('source', '알수없음')}")
            print(f"{doc.page_content[:150]}...\n")