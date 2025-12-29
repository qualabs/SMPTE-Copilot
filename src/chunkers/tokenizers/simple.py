from __future__ import annotations

"""Simple approximation tokenizer implementation."""

from typing import Any

from pydantic import ConfigDict, Field

from .base_tokenizer import Tokenizer

class SimpleTokenizer(Tokenizer):
    """Simple tokenizer that uses character-based approximation for token counting.
    
    This tokenizer doesn't require any external API and works independently
    of the embedding model. It uses a fast approximation (4 chars per token).
    """

    model_config = ConfigDict(extra='allow')

    def __init__(self, max_tokens: int = 2048, **kwargs):
        """Initialize the approximation tokenizer.

        Parameters
        ----------
        max_tokens
            Maximum number of tokens per chunk (default: 2048).
        """
        super().__init__(**kwargs)
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

    Returns
    -------
    Tokenizer instance.
    """
    max_tokens = config.get("max_tokens", 2048)
    return SimpleTokenizer(max_tokens=max_tokens)

