import torch
import math
import time
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

# [디버그용 로깅 설정] 터미널에 보기 좋게 색상과 시간을 찍어줍니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Reranker_Debug")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    """
    [기능] 랭킹 모델을 메모리에 로드하며, 각 단계별로 디버그 로그를 출력합니다.
    """
    global _tokenizer, _model
    if _model is None:
        logger.info("="*60)
        logger.info(f"🚀 [디버그 시작] 랭킹 모델 로드 프로세스 가동: {config.RERANK_MODEL}")
        logger.info(f"💻 현재 할당된 디바이스: {_device}")
        
        # 다운로드되는 파일이 저장되는 디스크 위치 확인
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        logger.info(f"📂 허깅페이스 캐시(다운로드) 저장 경로: {cache_dir}")
        logger.info("="*60)

        try:
            # --- [Step 1] 토크나이저 로드 ---
            start_time = time.time()
            logger.info("⏳ [Step 1] 토크나이저(Tokenizer) 다운로드 및 로드 시도 중...")
            _tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
            logger.info(f"✅ [Step 1 완료] 토크나이저 로드 성공! (소요 시간: {time.time() - start_time:.2f}초)")

            # --- [Step 2] 모델 가중치 로드 (가장 위험한 구간!) ---
            start_time = time.time()
            logger.info("⏳ [Step 2] 모델 가중치(Model Weights) 다운로드 및 로드 시도 중...")
            logger.info("   ⚠️ 주의: 여기서 터미널이 멈추거나 튕긴다면 (1) RAM 메모리 부족 (2) 디스크 용량 부족 (3) 네트워크 타임아웃 중 하나입니다!")
            _model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL)
            logger.info(f"✅ [Step 2 완료] 모델 가중치 로드 성공! (소요 시간: {time.time() - start_time:.2f}초)")

            # --- [Step 3] 디바이스(CPU/GPU) 할당 ---
            start_time = time.time()
            logger.info(f"⏳ [Step 3] 모델을 {_device} 메모리로 이동 중...")
            _model = _model.to(_device)
            _model.eval()
            logger.info(f"✅ [Step 3 완료] 디바이스 할당 및 추론 모드(eval) 세팅 성공! (소요 시간: {time.time() - start_time:.2f}초)")
            logger.info("="*60)

        except Exception as e:
            logger.error("❌ [치명적 에러 발생] 모델 로드 중 파이썬 프로세스가 예외를 던졌습니다!")
            logger.error(f"에러 상세 내용: {e}")
            raise e

# (이하 rerank_and_score 함수는 기존과 동일하게 유지)
def rerank_and_score(query: str, candidates: list, top_n: int = 3):
    if not candidates: return []
    _load_model() 

    scored_docs = []
    for doc in candidates:
        inputs = _tokenizer(query, doc.page_content, return_tensors="pt", truncation=True, max_length=512).to(_device)
        with torch.no_grad():
            logit = _model(**inputs).logits[0][0].item()
            prob = 1 / (1 + math.exp(-logit))
        scored_docs.append((prob, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs[:top_n]