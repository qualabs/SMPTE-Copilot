"""Document loader configuration."""

from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class FileTypeLoaderConfig(BaseSettings):
    """Configuration for a specific file type loader."""

    loader_name: str = Field(
        description="Name of the loader to use (e.g., 'pymupdf', 'video_loader')"
    )
    loader_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Loader-specific configuration dictionary"
    )


class LoaderConfig(BaseSettings):
    """Document loader configuration."""

    file_type_mapping: dict[str, dict[str, Any]] = Field(
        description=(
            "Mapping of file extensions to loader configuration. "
            "Each entry must be an object with 'loader_name' and optional 'loader_config'. "
            "Format: {'.pdf': {'loader_name': 'pymupdf', 'loader_config': {...}}, ...}"
        ),
    )

