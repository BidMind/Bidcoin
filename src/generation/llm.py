# OpenAI API를 사용하여 텍스트를 생성하는 클라이언트 클래스. 
# .env 파일에서 설정값을 읽어와서 OpenAI API 클라이언트를 초기화하고, 
# 주어진 지침과 사용자 입력에 따라 텍스트를 생성하는 메서드를 제공합니다.

from __future__ import annotations

from openai import OpenAI

from .config import Settings


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate_text(self, instructions: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.settings.model,
            instructions=instructions,
            input=user_input,
            reasoning={"effort": self.settings.reasoning_effort},
        )
        return response.output_text
