import os
from dotenv import load_dotenv
from openai import OpenAI

from .base_client import BaseClient


class OpenRouterClient(BaseClient):
    MODEL = "qwen/qwen3-32b"
    BASE_URL = "https://openrouter.ai/api/v1"
    RPM = 30

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        rpm: int | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()
        key = api_key or os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            key,
            base_url or self.BASE_URL,
            model_name or self.MODEL,
            rpm or self.RPM,
        )
