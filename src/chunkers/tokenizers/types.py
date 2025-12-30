"""Tokenizer types."""

from enum import Enum


class TokenizerType(str, Enum):
    """Tokenizer type enumeration."""

    SIMPLE = "simple"
    GEMINI = "gemini"

