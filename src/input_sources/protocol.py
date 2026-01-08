from __future__ import annotations

"""Protocol for input source implementations."""

from pathlib import Path
from typing import Protocol


class InputSource(Protocol):
    """Protocol for input source implementations.

    Defines the interface for accessing files from different sources
    (local filesystem, S3, etc.). All implementations must provide
    these methods to be compatible.
    """

    def list_files(self, path: str, extensions: list[str] | None = None) -> list[str]:
        """List files in the given path.

        Parameters
        ----------
        path
            Path to list files from (can be directory path or S3 prefix).
        extensions
            Optional list of file extensions to filter by (e.g., ['.pdf', '.docx']).
            If None, return all files.

        Returns
        -------
        List of file identifiers (paths or URIs) that can be used with get_file().
        """
        ...

    def get_file(self, file_id: str) -> Path:
        """Get a file and return a local path to it.

        For local sources, this returns the path directly.
        For remote sources (S3), this downloads the file to a temp location.

        Parameters
        ----------
        file_id
            File identifier (local path or S3 URI).

        Returns
        -------
        Path to the local file (original or downloaded temp file).
        """
        ...

    def cleanup(self) -> None:
        """Clean up any temporary files or resources.

        Called after processing is complete to remove downloaded files,
        close connections, etc.
        """
        ...
