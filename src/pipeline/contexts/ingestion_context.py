"""Context for document ingestion pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document

from ..context import PipelineContext


class IngestionContext(PipelineContext):
    """Context for document ingestion pipeline.

    Tracks the state of a document as it moves through the ingestion pipeline:
    Load -> Chunk -> Embed -> Save
    """

    file_path: Path
    raw_text: Optional[str] = None
    markdown_path: Optional[Path] = None
    chunks: list[Document] = []
    vectors: list[list[float]] = []
    metadata: dict[str, Any] = {}
    
    # Role-aware access control fields (optional)
    access_metadata: dict[str, Any] = {}  # Additional metadata for access control
    access_tags: list[str] = []  # Tags for document-level access control
    required_role_strict: Optional[str] = None  # Strict role requirement for document access
