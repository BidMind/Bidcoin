# BidCoinGenerator 클래스는 검색된 컨텍스트와 대화 이력을 바탕으로 LLM을 사용하여 답변을 생성하는 역할을 합니다.

from __future__ import annotations

from .config import Settings
from .context_builder import build_context_block, build_history_block
from .llm import OpenAIClient
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import GenerationResponse, RetrievalResult


class BidCoinGenerator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = OpenAIClient(self.settings)

    def generate(self, retrieval_result: RetrievalResult) -> GenerationResponse:
        context_block, used_sources = build_context_block(
            contexts=retrieval_result.contexts,
            max_contexts=self.settings.max_contexts,
            max_chars=self.settings.max_context_chars,
        )
        history_block = build_history_block(retrieval_result.chat_history)
        user_prompt = build_user_prompt(
            question=retrieval_result.question,
            context_block=context_block,
            history_block=history_block,
        )

        answer = self.llm.generate_text(
            instructions=SYSTEM_PROMPT,
            user_input=user_prompt,
        )

        return GenerationResponse(
            answer=answer,
            used_context_count=min(len(retrieval_result.contexts), self.settings.max_contexts),
            used_sources=used_sources,
            context_preview=context_block[:1000],
            raw_model_output=answer,
        )
