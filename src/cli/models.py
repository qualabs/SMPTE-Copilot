"""Types and result classes for CLI operations."""

from pathlib import Path
from typing import Optional

from src.config import Config
from src.embeddings import Embeddings
from src.input_sources import InputSourceType
from src.vector_stores import VectorStore


class IngestionConfig:
    """Configuration for ingestion operations."""

    def __init__(
        self,
        source_type: InputSourceType,
        source_config: dict,
        config: Config,
        source_ids: Optional[list[str]] = None,
        embedding_model: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize the ingestion configuration.

        Parameters
        ----------
        source_type
            Type of input source (LOCAL or S3).
        source_config
            Configuration dictionary for the input source.
        config
            Configuration object.
        source_ids
            Optional list of source identifiers (S3 URIs or local paths) to process.
        embedding_model
            Embedding model instance (can be None if save step is disabled).
        vector_store
            Vector store instance (can be None if save step is disabled).
        """
        self.source_type = source_type
        self.source_config = source_config
        self.config = config
        self.source_ids = source_ids
        self.embedding_model = embedding_model
        self.vector_store = vector_store


class IngestionResult:
    """Result of a single file ingestion."""

    def __init__(
        self,
        file_path: Path,
        success: bool,
        error: Optional[str] = None,
        chunks_count: int = 0,
        markdown_path: Optional[Path] = None,
    ):
        """Initialize the ingestion result.

        Parameters
        ----------
        file_path
            Path to the ingested file.
        success
            Whether the ingestion was successful.
        error
            Error message if ingestion failed.
        chunks_count
            Number of chunks created.
        markdown_path
            Path to the generated markdown file.
        """
        self.file_path = file_path
        self.success = success
        self.error = error
        self.chunks_count = chunks_count
        self.markdown_path = markdown_path

