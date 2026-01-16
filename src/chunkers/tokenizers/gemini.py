from __future__ import annotations

"""Gemini tokenizer implementation."""

from typing import Any

from google import genai
from pydantic import ConfigDict

from .base_tokenizer import Tokenizer


class GeminiTokenizer(Tokenizer):
    """Custom tokenizer for Docling's HybridChunker that uses Gemini's token counting API.

    This tokenizer integrates with Google's Gemini API to accurately count tokens
    for text chunks, with a fast local fallback for performance.
    """

    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        max_tokens: int = 2048,
        google_api_key: str | None = None,
        chars_per_token_ratio: float | None = None,
        split_buffer_size: int | None = None,
        **kwargs
    ):
        """Initialize the Gemini tokenizer.

        Parameters
        ----------
        model
            Gemini model name to use for token counting (default: gemini-embedding-001).
            Note: Some models like "models/embedding-001" don't support countTokens API.
        max_tokens
            Maximum number of tokens per chunk (default: 2048).
        google_api_key
            Optional Google API key. Used only if client is not provided.
            If not provided, uses GOOGLE_API_KEY env var.
        chars_per_token_ratio
            Ratio of characters to tokens for threshold estimation.
        split_buffer_size
            Number of words to buffer before checking limits.
        """
        super().__init__(
            chars_per_token_ratio=chars_per_token_ratio,
            split_buffer_size=split_buffer_size,
            **kwargs
        )
        self.client = genai.Client(api_key=google_api_key)
        self.model = model
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's API with fallback to local estimation.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        Number of tokens (from API or estimated).
        """
        response = self.client.models.count_tokens(model=self.model, contents=text)
        return response.total_tokens

    def get_max_tokens(self) -> int:
        """Returns the maximum tokens allowed per chunk.

        Returns
        -------
        Maximum tokens.
        """
        return self.max_tokens

    def _hash_attributes(self) -> tuple:
        """Return hashable attributes that uniquely identify this tokenizer.

        Returns
        -------
        Tuple containing class type, model name, and configuration attributes.
        Note: Does not include client or google_api_key as these are implementation
        details, not part of the tokenizer's identity.
        """
        return (
            type(self),
            self.model,
            self.max_tokens,
            self.chars_per_token_ratio,
            self.split_buffer_size,
        )

def create_gemini_tokenizer(config: dict[str, Any]) -> Tokenizer:
    """Create a Gemini tokenizer from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - max_tokens: int (optional) - Maximum tokens per chunk (default: 2048)
        - google_api_key: str (optional) - Google API key for token counting
        - model: str (optional) - Gemini model name for token counting (default: gemini-embedding-001)
        - chars_per_token_ratio: float (optional) - Char-to-token ratio for threshold (default: 1.5)
        - split_buffer_size: int (optional) - Words to buffer before checking limits (default: 5)

    Returns
    -------
    Tokenizer instance.

    Raises
    ------
    ValueError
        If invalid configuration values are provided.
    """
    max_tokens = config.get("max_tokens", 2048)
    google_api_key = config.get("llm_api_key")
    tokenizer_model = config.get("llm_model", "gemini-embedding-001")
    chars_per_token_ratio = config.get("chars_per_token_ratio")
    split_buffer_size = config.get("split_buffer_size")

    return GeminiTokenizer(
        model=tokenizer_model,
        max_tokens=max_tokens,
        google_api_key=google_api_key,
        chars_per_token_ratio=chars_per_token_ratio,
        split_buffer_size=split_buffer_size,
    )

