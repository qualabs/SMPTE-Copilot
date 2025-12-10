"""Retrieval pipeline configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseSettings):
    """Retrieval pipeline configuration."""
    
    searcher_strategy: str = Field(
        default="similarity",
        description="Retrieval strategy (similarity, etc.)",
    )
    k: int = Field(
        default=5,
        description="Number of results to retrieve",
        gt=0,
    )
    searcher_config: Optional[dict] = Field(
        default=None,
        description="Additional searcher-specific configuration",
    )

