"""Document loader configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from src.loaders.types import LoaderType


class LoaderConfig(BaseSettings):
    """Document loader configuration."""
    
    loader_name: LoaderType = Field(
        default=LoaderType.PYMUPDF,
        description="Document loader type",
    )
    loader_config: Optional[dict] = Field(
        default=None,
        description="Additional loader-specific configuration (dict)",
    )

