"""Gemini tokenizer implementation."""
import logging
import os

from __future__ import annotations

from typing import Any, Optional

from pydantic import ConfigDict

from .protocol import Tokenizer

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from google import genai


class GeminiTokenizer(Tokenizer):
    """Custom tokenizer for Docling's HybridChunker that uses Gemini's token counting API.

    This tokenizer integrates with Google's Gemini API to accurately count tokens
    for text chunks, with a fast local fallback for performance.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        max_tokens: int = 2048,
        google_api_key: Optional[str] = None,
    ):
        """Initialize the Gemini tokenizer.

        Parameters
        ----------
        client
            Optional genai.Client instance. If not provided, will be created from API key.
        model
            Gemini model name to use for token counting (default: gemini-embedding-001).
            Note: Some models like "models/embedding-001" don't support countTokens API.
        max_tokens
            Maximum number of tokens per chunk (default: 2048).
        google_api_key
            Optional Google API key. Used only if client is not provided.
            If not provided, uses GOOGLE_API_KEY env var.
        """

        self.client = genai.Client(api_key=google_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger()
                
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
        return self.client.models.count_tokens(model=self.model, contents=text)

    def get_max_tokens(self) -> int:
        """Returns the maximum tokens allowed per chunk.

        Returns
        -------
        Maximum tokens.
        """
        return self.max_tokens

def create_gemini_tokenizer(config: dict[str, Any]) -> Tokenizer:
    """Create a Gemini tokenizer from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - max_tokens: int (optional) - Maximum tokens per chunk (default: 2048)
        - google_api_key: str (optional) - Google API key for token counting
        - model: str (optional) - Gemini model name for token counting (default: gemini-embedding-001)

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

    return GeminiTokenizer(
        model=tokenizer_model,
        max_tokens=max_tokens,
        google_api_key=google_api_key
    )

