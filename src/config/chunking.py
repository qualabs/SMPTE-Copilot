"""Chunking configuration."""

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from src.chunkers.types import ChunkerType
from src.chunkers.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNKING_METHOD_RECURSIVE,
)


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
        description="Chunking method to use",
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

