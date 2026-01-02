"""Document loader configuration."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class LoaderConfig(BaseSettings):
    """Document loader configuration."""

    file_type_mapping: list[dict[str, Any]] = Field(
        description=(
            "Mapping of file extensions to loader configuration. "
            "List where each entry has 'extensions' (list of extensions), "
            "'loader_name', and optional 'loader_config'. "
            "Format: [{'extensions': ['.pdf', '.docx'], 'loader_name': '...', 'loader_config': {...}}, ...]"
        ),
    )

