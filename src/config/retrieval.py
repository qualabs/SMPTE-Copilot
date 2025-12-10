"""Retrieval pipeline configuration."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.constants import DEFAULT_RETRIEVAL_K  # Shared across retrievers, vector_stores, and config
from src.retrievers.types import RetrieverType


class RetrievalConfig(BaseSettings):
    """Retrieval pipeline configuration."""

    searcher_strategy: RetrieverType = Field(
        default=RetrieverType.SIMILARITY,
        description="Retrieval strategy type",
    )
    k: int = Field(
        default=DEFAULT_RETRIEVAL_K,
        description="Number of results to retrieve",
        gt=0,
    )
    searcher_config: Optional[dict] = Field(
        default=None,
        description="Additional searcher-specific configuration",
    )

