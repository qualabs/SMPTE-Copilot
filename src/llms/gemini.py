from __future__ import annotations

from typing import Any, Dict, Optional

from google import genai

from .protocol import LLM


class GeminiLLM(LLM):
    """LLM backed by Gemini API (Google GenAI SDK)."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return resp.text or ""


def create_gemini_llm(config: Dict[str, Any]) -> LLM:
    return GeminiLLM(
        model=config.get("model", "gemini-2.5-flash"),
        api_key=config.get("api_key"),
    )
