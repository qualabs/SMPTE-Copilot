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
