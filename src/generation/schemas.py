# 입력/출력 형태 정의.


from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ChatTurn: 대화 이력의 한 턴을 나타냄. 역할과 내용을 포함
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="대화 역할")
    content: str = Field(..., description="대화 내용")

# RetrievedContext: 검색된 문서 청크를 나타냄.
class RetrievedContext(BaseModel):
    chunk_id: str | None = Field(default=None, description="청크 식별자")
    text: str = Field(..., min_length=1, description="retrieval로 찾은 청크 본문")
    source_file: str = Field(..., min_length=1, description="출처 파일명")
    organization: str | None = Field(default=None, description="발주기관")
    project_name: str | None = Field(default=None, description="사업명")
    summary: str | None = Field(default=None, description="문서 요약")
    score: float | None = Field(default=None, description="retrieval 유사도 점수")

# RetrievalResult: 검색 결과 전체를 나타냄
class RetrievalResult(BaseModel):
    question: str = Field(..., min_length=1)
    contexts: list[RetrievedContext] = Field(default_factory=list)
    chat_history: list[ChatTurn] = Field(default_factory=list)

# GenerationResponse: 답변 생성 결과를 나타냄. 생성된 답변, 사용된 컨텍스트 수, 사용된 출처 목록, 컨텍스트 미리보기, 원시 모델 출력 등을 포함
class GenerationResponse(BaseModel):
    answer: str
    used_context_count: int
    used_sources: list[str]
    context_preview: str
    raw_model_output: str | None = None
