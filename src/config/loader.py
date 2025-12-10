"""Document loader configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class LoaderConfig(BaseSettings):
    """Document loader configuration."""
    
    loader_name: str = Field(
        default="pymupdf",
        description="Document loader name (pymupdf)",
    )
    loader_config: Optional[dict] = Field(
        default=None,
        description="Additional loader-specific configuration (dict)",
    )

