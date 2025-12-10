"""Chunking configuration."""

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from src.chunkers.types import ChunkerType


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""
    
    chunker_name: ChunkerType = Field(
        default=ChunkerType.LANGCHAIN,
        description="Chunker type",
    )
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks in characters",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters",
        ge=0,
    )
    method: Literal["recursive", "character", "token"] = Field(
        default="recursive",
        description="Chunking method to use",
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

