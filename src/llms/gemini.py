from __future__ import annotations

from typing import Any

from google import genai

from .protocol import LLM


class GeminiLLM(LLM):
    """LLM backed by Gemini API (Google GenAI SDK)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
    ):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> str:
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return resp.text or ""


def create_gemini_llm(config: dict[str, Any]) -> LLM:
    return GeminiLLM(
        model=config.get("model", "gemini-2.5-flash"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0.3),
        max_output_tokens=config.get("max_output_tokens", 2048),
    )
