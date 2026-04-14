import torch
import math
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

# [디버그용 로깅 설정] 터미널에 시각적으로 보기 좋게 출력합니다.
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Reranker_Debug")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    """
    [기능] 랭킹 모델을 메모리에 로드합니다.
    """
    global _tokenizer, _model
    if _model is None:
        logger.info("\n" + "="*60)
        logger.info(f"🚀 랭킹 모델 로드 중: {config.RERANK_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL).to(_device)
        _model.eval()
        logger.info("✅ 랭킹 모델 로드 완료!")
        logger.info("="*60 + "\n")

def rerank_and_score(query: str, candidates: list, top_n: int = 3):
    """
    [기능] 1차 후보군을 채점하며, 0점이 나오는 원인을 터미널에 출력합니다.
    """
    if not candidates: return []
    _load_model()

    scored_docs = []
    
    logger.info(f"🔍 [Reranker 채점 시작]")
    logger.info(f"🧐 기준 질문(Query): '{query}'")
    logger.info(f"📚 후보 문서 수: {len(candidates)}개\n" + "-"*60)

    for i, doc in enumerate(candidates):
        inputs = _tokenizer(query, doc.page_content, return_tensors="pt", truncation=True, max_length=512).to(_device)
        
        with torch.no_grad():
            logit = _model(**inputs).logits[0][0].item()
            prob = 1 / (1 + math.exp(-logit))
            
            # --- 💡 [디버그 CCTV 영역] 터미널에 상세 내역 출력 ---
            # 줄바꿈 문자를 공백으로 치환하여 터미널 출력이 깨지지 않게 방어
            preview_text = doc.page_content[:100].replace('\n', ' ')
            
            logger.info(f"[후보 {i+1}] 미리보기: {preview_text}...")
            logger.info(f"   ⚖️ Raw 로짓(Logit): {logit:.4f}  ➡️  Sigmoid 확률: {prob:.4f}")
            
            if prob < 0.01:
                logger.warning("   ⚠️ [경고] 확률이 0에 수렴합니다! (관련 없는 문서이거나, 핵심 내용이 512토큰 뒤에 잘렸습니다.)")
            logger.info("-" * 60)
            # --------------------------------------------------

        scored_docs.append((prob, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    logger.info(f"🏆 [채점 완료] 1위 문서 최종 점수: {scored_docs[0][0]:.4f}\n" + "="*60)
    return scored_docs[:top_n]