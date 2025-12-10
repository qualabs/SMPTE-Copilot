"""Chunker types."""
from enum import Enum


class ChunkerType(str, Enum):
    """Chunker type enumeration."""

    LANGCHAIN = "langchain"

