"""File paths configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class PathsConfig(BaseSettings):
    """File paths configuration."""
    
    input_path: Path = Field(
        default=Path("./data"),
        description="Default path for input media files (relative to current working directory)",
    )
    markdown_dir: Path = Field(
        default=Path("./data/markdown"),
        description="Directory for markdown output (relative to current working directory)",
    )

