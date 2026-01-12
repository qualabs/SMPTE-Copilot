from __future__ import annotations

"""Context for document ingestion pipeline."""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from ..context import PipelineContext


class IngestionContext(PipelineContext):
    """Context for document ingestion pipeline.

    Tracks the state of a document as it moves through the ingestion pipeline:
    Load -> Chunk -> Embed -> Save
    """

    file_path: Path
    raw_text: str | None = None
    markdown_path: Path | None = None
    chunks: list[Document] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Metadata extracted from the loaded document

    access_metadata: dict[str, Any] = Field(default_factory=dict)  # Additional metadata for access control
    access_tags: list[str] = Field(default_factory=list)  # Tags for document-level access control
