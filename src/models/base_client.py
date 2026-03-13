from typing import Any
from openai import OpenAI

from .rate_limiter import RateLimiter


class BaseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        rpm: int,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        self._openai = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.base_url = base_url
        self._rate_limiter = RateLimiter(rpm)

    def chat_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ):
        self._rate_limiter.acquire()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self._openai.chat.completions.create(**kwargs)
