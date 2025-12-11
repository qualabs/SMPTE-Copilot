"""Helper class for working with loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from .constants import SUPPORTED_FILE_EXTENSIONS


class LoaderHelper:
    """Static helper class for loader operations."""

    @staticmethod
    def get_loader_config_for_file(
        file_path: Path,
        config: Config,
    ) -> tuple[str, dict[str, Any]]:
        """Get the loader name and config for a file based on its extension and configuration.

        Parameters
        ----------
        file_path
            Path to the file.
        config
            Configuration object.

        Returns
        -------
        Tuple of (loader_name, loader_config_dict).

        Raises
        ------
        ValueError
            If the file type is not supported, no loader is configured for it,
            or the loader configuration is invalid.
        """
        if not isinstance(file_path, Path):
            raise TypeError(f"file_path must be a Path object, got {type(file_path)}")

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_FILE_EXTENSIONS:
            supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {supported}"
            )

        if suffix not in config.loader.file_type_mapping:
            raise ValueError(
                f"No loader configured for file type: {suffix}. "
                f"Please add '{suffix}: {{loader_name: ..., loader_config: ...}}' "
                f"to loader.file_type_mapping in config.yaml"
            )

        loader_entry = config.loader.file_type_mapping[suffix]

        if not isinstance(loader_entry, dict):
            raise TypeError(
                f"Loader entry for '{suffix}' must be a dictionary with 'loader_name' key. "
                f"Received: {type(loader_entry)}"
            )

        loader_name = loader_entry.get("loader_name")
        if not loader_name:
            raise ValueError(
                f"Loader entry for '{suffix}' must have 'loader_name' key. "
                f"Received: {loader_entry}"
            )

        if not isinstance(loader_name, str) or not loader_name.strip():
            raise ValueError(
                f"Loader name for '{suffix}' must be a non-empty string. "
                f"Received: {loader_name!r}"
            )

        loader_config = loader_entry.get("loader_config") or {}
        if not isinstance(loader_config, dict):
            raise TypeError(
                f"Loader config for '{suffix}' must be a dictionary or None. "
                f"Received: {type(loader_config)}"
            )

        return loader_name, loader_config

    @staticmethod
    def resolve_media_inputs(input_path: Path) -> list[Path]:
        """Resolve input path to a list of supported media files.

        Parameters
        ----------
        input_path
            Path to a file or directory.

        Returns
        -------
        List of file paths (sorted).

        Raises
        ------
        TypeError
            If input_path is not a Path object.
        FileNotFoundError
            If the path doesn't exist or no supported files are found.
        ValueError
            If a file has an unsupported extension.
        """
        if not isinstance(input_path, Path):
            raise TypeError(f"input_path must be a Path object, got {type(input_path)}")

        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")

        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix not in SUPPORTED_FILE_EXTENSIONS:
                supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
                raise ValueError(
                    f"Unsupported file type: {suffix}. "
                    f"Supported types: {supported}"
                )
            return [input_path]

        if not input_path.is_dir():
            raise ValueError(f"Path is neither a file nor a directory: {input_path}")

        media_files = [
            path for ext in SUPPORTED_FILE_EXTENSIONS
            for path in input_path.glob(f"*{ext}")
            if path.is_file()
        ]

        media_files = sorted(media_files)
        if not media_files:
            supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
            raise FileNotFoundError(
                f"No supported files found in directory: {input_path}. "
                f"Supported types: {supported}"
            )
        return media_files

    @staticmethod
    def prepare_output_dir(output_dir: Path) -> Path:
        """Prepare output directory, creating it if it doesn't exist.

        Parameters
        ----------
        output_dir
            Path to the output directory.

        Returns
        -------
        Path to the prepared output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def create_loader_config(
        file_path: Path,
        loader_name: str,
        loader_config_from_mapping: dict[str, Any],
        config: Config,
    ) -> dict[str, Any]:
        """Create loader configuration based on file type and loader name.

        Parameters
        ----------
        file_path
            Path to the media file.
        loader_name
            Name of the loader to use (e.g., "pymupdf", "video_loader").
        loader_config_from_mapping
            Loader-specific configuration from file_type_mapping.
        config
            Configuration object.

        Returns
        -------
        Configuration dictionary for the loader.

        Raises
        ------
        TypeError
            If any parameter has an invalid type.
        ValueError
            If loader_name is empty or loader_config_from_mapping is not a dict.
        """
        if not isinstance(file_path, Path):
            raise TypeError(f"file_path must be a Path object, got {type(file_path)}")

        if not isinstance(loader_name, str) or not loader_name.strip():
            raise ValueError(
                f"loader_name must be a non-empty string, got: {loader_name!r}"
            )

        if not isinstance(loader_config_from_mapping, dict):
            raise TypeError(
                f"loader_config_from_mapping must be a dict, got {type(loader_config_from_mapping)}"
            )

        output_dir = LoaderHelper.prepare_output_dir(config.paths.markdown_dir)

        result = loader_config_from_mapping.copy()
        result["file_path"] = str(file_path)
        result["output_dir"] = str(output_dir)

        return result

