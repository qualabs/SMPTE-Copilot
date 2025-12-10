"""Embedding model configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from src.embeddings.types import EmbeddingModelType


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    embed_name: EmbeddingModelType = Field(
        default=EmbeddingModelType.HUGGINGFACE,
        description="Embedding model type",
    )
    embed_config: Optional[dict] = Field(
        default=None,
        description="Additional model-specific keyword arguments",
    )

