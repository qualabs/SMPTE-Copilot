"""Chunking configuration."""

from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from src.chunkers.constants import (
    CHUNKING_METHOD_RECURSIVE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from src.chunkers.types import ChunkerType


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    chunker_name: ChunkerType = Field(
        default=ChunkerType.LANGCHAIN,
        description="Chunker type",
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="Size of text chunks in characters",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="Overlap between chunks in characters",
        ge=0,
    )
    method: Literal["recursive", "character", "token"] = Field(
        default=CHUNKING_METHOD_RECURSIVE,
        description="Chunking method to use (for langchain chunker)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens per chunk (for hybrid chunker, default: 2000)",
        gt=0,
    )
    merge_peers: bool = Field(
        default=False,
        description="Whether to merge peer chunks (for hybrid chunker)",
    )

    @model_validator(mode='after')
    def validate_overlap_less_than_size(self) -> 'ChunkingConfig':
        # Only validate chunk_size/chunk_overlap for langchain chunker
        # Hybrid chunker uses max_tokens (token-based), not chunk_size/overlap
        if self.chunker_name.value != "hybrid" and self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

