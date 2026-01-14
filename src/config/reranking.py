"""Reranking configuration."""


from pydantic import Field
from pydantic_settings import BaseSettings

from src.rerankers.types import RerankerType


class RerankingConfig(BaseSettings):
    """Reranking configuration."""

    reranker_name: RerankerType = Field(
        default=RerankerType.GEMINI,
        description="Reranker backend type",
    )
    reranker_config: dict | None = Field(
        default=None,
        description="Additional reranker-specific keyword arguments",
    )
