"""File paths configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class PathsConfig(BaseSettings):
    """File paths configuration."""
    
    pdf_path: Path = Field(
        default=Path("/app/data"),
        description="Default path for PDF files",
    )
    markdown_dir: Path = Field(
        default=Path("/app/data/markdown"),
        description="Directory for markdown output",
    )

