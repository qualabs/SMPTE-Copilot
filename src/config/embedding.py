"""Embedding model configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(
        default="huggingface",
        description="Embedding model name (huggingface, openai, sentence-transformers)",
    )
    model_kwargs: Optional[dict] = Field(
        default=None,
        description="Additional model-specific keyword arguments",
    )

