import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

# 랭킹 모델과 토크나이저를 전역 변수로 관리하여 메모리 효율적으로 사용
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    """
    [기능] 랭킹 모델을 메모리에 로드합니다. 모델이 이미 로드되어 있다면 재사용합니다.
    """
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL).to(_device)
        _model.eval() # 추론 모드로 고정

def rerank_and_score(query: str, candidates: list, top_n: int = 3):
    """
    [기능] 1차 후보군을 질문과 1:1로 비교 채점하여 0~1 사이의 확률 점수로 랭킹합니다.
    """
    if not candidates: return []
    _load_model() # 모델 준비 확인

    scored_docs = []
    # Cross-Encoder 채점
    for doc in candidates:
        # 질문과 문서을 토크나이저에 동시에 넣어서 하나의 텐서로 만듬
        inputs = _tokenizer(query, doc.page_content, return_tensors="pt", truncation=True, max_length=512).to(_device)
        
        with torch.no_grad(): # 역전파 연산을 끄고 속도를 높임
            # 모델이 BGE 특유의 로짓(Logit) 점수를 반환 (예: 4.8, 2.3, -1.5 등)
            logit = _model(**inputs).logits[0][0].item()

            # Logit 점수를 0~1 사이의 확률로 변환 (예: 0.99, 0.91, 0.18 등)
            prob = 1 / (1 + math.exp(-logit))  # 0~1 사이의 확률로 변환
        scored_docs.append((prob, doc)) # (점수, 문서) 튜플로 저장

    scored_docs.sort(key=lambda x: x[0], reverse=True)  # 점수 기준으로 정렬
    return scored_docs[:top_n]