from __future__ import annotations

"""Local filesystem input source implementation."""

import logging
from pathlib import Path
from typing import Any

from .protocol import InputSource


class LocalInputSource:
    """Input source for local filesystem.

    Provides access to files on the local filesystem without any
    downloading or temporary file handling.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the local input source.

        Parameters
        ----------
        config
            Configuration dictionary (currently unused for local source).
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def list_files(self, path: str, extensions: list[str] | None = None) -> list[str]:
        """List files in the given local path.

        Parameters
        ----------
        path
            Local directory path to list files from.
        extensions
            Optional list of file extensions to filter by (e.g., ['.pdf', '.docx']).

        Returns
        -------
        List of file paths as strings.
        """
        local_path = Path(path).expanduser().resolve()

        if not local_path.exists():
            raise FileNotFoundError(f"Path does not exist: {local_path}")

        if local_path.is_file():
            # Single file
            if extensions is None or local_path.suffix.lower() in extensions:
                return [str(local_path)]
            return []

        # Directory - recursively find all files
        files = []
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    files.append(str(file_path))

        self.logger.info(f"Found {len(files)} file(s) in {local_path}")
        return sorted(files)

    def get_file(self, file_id: str) -> Path:
        """Get a local file path.

        For local sources, this simply validates and returns the path.

        Parameters
        ----------
        file_id
            Local file path.

        Returns
        -------
        Path object pointing to the file.
        """
        local_path = Path(file_id).expanduser().resolve()

        if not local_path.exists():
            raise FileNotFoundError(f"File does not exist: {local_path}")

        if not local_path.is_file():
            raise ValueError(f"Path is not a file: {local_path}")

        return local_path

    def cleanup(self) -> None:
        """Clean up resources.

        For local source, nothing to clean up.
        """
        pass


def create_local_source(config: dict[str, Any]) -> InputSource:
    """Create a local filesystem input source.

    Parameters
    ----------
    config
        Configuration dictionary (currently unused).

    Returns
    -------
    InputSource instance for local filesystem.
    """
    return LocalInputSource(config)
