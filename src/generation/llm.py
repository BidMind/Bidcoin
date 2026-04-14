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
            temperature=self.settings.temperature,
        )
        return response.output_text