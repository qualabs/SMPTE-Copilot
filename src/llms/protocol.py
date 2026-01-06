from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    """Protocol for LLM chat/generation backends."""
    def generate(self, prompt: str) -> str:
        """Generate a response from a plain prompt string."""
        ...
