from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "low")
    # temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    max_contexts: int = int(os.getenv("MAX_CONTEXTS", "3"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 또는 환경변수를 확인하세요."
            )
        if self.reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError(
                "OPENAI_REASONING_EFFORT는 none/low/medium/high 중 하나여야 합니다."
            )
        # if not (0.0 <= self.temperature <= 2.0):
        #     raise ValueError("OPENAI_TEMPERATURE는 0.0 이상 2.0 이하여야 합니다.")