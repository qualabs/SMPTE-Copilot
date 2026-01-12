from __future__ import annotations

"""Context for document ingestion pipeline."""

from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from pydantic import Field

from ..context import PipelineContext


class IngestionContext(PipelineContext):
    """Context for document ingestion pipeline.

    Tracks the state of a document as it moves through the ingestion pipeline:
    Load -> Chunk -> Embed -> Save
    """

    source_id: str  # Original source identifier (S3 URI or local path)
    file_path: Path  # Resolved local file path for processing
    raw_text: Optional[str] = None
    markdown_path: Optional[Path] = None
    chunks: list[Document] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Metadata extracted from the loaded document

    access_metadata: dict[str, Any] = Field(default_factory=dict)  # Additional metadata for access control
    access_tags: list[str] = Field(default_factory=list)  # Tags for document-level access control
