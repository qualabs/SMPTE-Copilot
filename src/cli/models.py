"""Types and result classes for CLI operations."""

from enum import Enum
from pathlib import Path


class ExecutorType(str, Enum):
    """Type of executor for parallel processing."""

    THREAD = "thread"
    PROCESS = "process"


class IngestionResult:
    """Result of a single file ingestion."""

    def __init__(
        self,
        file_path: Path,
        success: bool,
        error: str | None = None,
        chunks_count: int = 0,
        markdown_path: Path | None = None,
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

