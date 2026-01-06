"""Chunking configuration."""

from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.chunkers.types import ChunkerType


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    chunker_name: ChunkerType = Field(
        default=ChunkerType.LANGCHAIN,
        description="Chunker type",
    )
    chunker_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Chunker-specific configuration dictionary. "
        "For langchain: chunk_size, chunk_overlap, method. "
        "For hybrid: max_tokens, merge_peers, tokenizer, tokenizer_config.",
    )
