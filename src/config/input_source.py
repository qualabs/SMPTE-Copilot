"""Input source configuration."""

from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class InputSourceConfig(BaseSettings):
    """Configuration for input sources (local, S3, etc.)."""

    source_type: str = Field(
        default="local",
        description="Type of input source: 'local' or 's3'",
    )

    source_config: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Source-specific configuration",
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"
