from __future__ import annotations

"""Simple approximation tokenizer implementation."""

from typing import Any

from pydantic import ConfigDict

from .base_tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Simple tokenizer that uses character-based approximation for token counting.

    This tokenizer doesn't require any external API and works independently
    of the embedding model. It uses a fast approximation (4 chars per token).
    """

    model_config = ConfigDict(extra='allow')

    def __init__(
        self,
        max_tokens: int = 2048,
        chars_per_token_ratio: float | None = None,
        split_buffer_size: int | None = None,
        **kwargs
    ):
        """Initialize the approximation tokenizer.

        Parameters
        ----------
        max_tokens
            Maximum number of tokens per chunk (default: 2048).
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
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens using character-based approximation.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        Number of tokens (approximated as 4 characters per token).
        """
        return len(text) // 4

    def get_max_tokens(self) -> int:
        """Returns the maximum tokens allowed per chunk."""
        return self.max_tokens

def create_simple_tokenizer(config: dict[str, Any]) -> Tokenizer:
    """Create a simple approximation tokenizer from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - max_tokens: int (optional) - Maximum tokens per chunk (default: 2048)
        - chars_per_token_ratio: float (optional) - Char-to-token ratio for threshold (default: 1.5)
        - split_buffer_size: int (optional) - Words to buffer before checking limits (default: 5)

    Returns
    -------
    Tokenizer instance.
    """
    max_tokens = config.get("max_tokens", 2048)
    chars_per_token_ratio = config.get("chars_per_token_ratio")
    split_buffer_size = config.get("split_buffer_size")
    return SimpleTokenizer(
        max_tokens=max_tokens,
        chars_per_token_ratio=chars_per_token_ratio,
        split_buffer_size=split_buffer_size,
    )

