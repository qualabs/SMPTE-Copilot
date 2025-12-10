"""Helper class for working with loaders."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

from ..config import Config
from .constants import SUPPORTED_FILE_EXTENSIONS

class LoaderHelper:
    """Static helper class for loader operations."""
    
    @staticmethod
    def get_loader_config_for_file(
        file_path: Path,
        config: Config,
    ) -> Tuple[str, Dict[str, Any]]:
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
            If the file type is not supported or no loader is configured for it.
        """
        suffix = file_path.suffix.lower()
        
        # Check if extension is supported
        if suffix not in SUPPORTED_FILE_EXTENSIONS:
            supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {supported}"
            )
        
        # Get loader from file type mapping (required)
        if suffix not in config.loader.file_type_mapping:
            raise ValueError(
                f"No loader configured for file type: {suffix}. "
                f"Please add '{suffix}: {{loader_name: ..., loader_config: ...}}' "
                f"to loader.file_type_mapping in config.yaml"
            )
        
        loader_entry = config.loader.file_type_mapping[suffix]
        
        if not isinstance(loader_entry, dict):
            raise ValueError(
                f"Loader entry for '{suffix}' must be a dictionary with 'loader_name' key. "
                f"Received: {type(loader_entry)}"
            )
        
        loader_name = loader_entry.get("loader_name")
        if not loader_name:
            raise ValueError(
                f"Loader entry for '{suffix}' must have 'loader_name' key. "
                f"Received: {loader_entry}"
            )
        
        loader_config = loader_entry.get("loader_config") or {}
        return loader_name, loader_config

    @staticmethod
    def resolve_media_inputs(input_path: Path) -> List[Path]:
        """Resolve input path to a list of supported media files.
        
        Parameters
        ----------
        input_path
            Path to a file or directory.
            
        Returns
        -------
        List of file paths.
            
        Raises
        ------
        FileNotFoundError
            If the path doesn't exist or no supported files are found.
        ValueError
            If a file has an unsupported extension.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")
        
        if input_path.is_file():
            # Validate file type
            suffix = input_path.suffix.lower()
            if suffix not in SUPPORTED_FILE_EXTENSIONS:
                supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
                raise ValueError(
                    f"Unsupported file type: {suffix}. "
                    f"Supported types: {supported}"
                )
            return [input_path]
        
        # Directory: collect all supported files (non-recursive)
        media_files = []
        for ext in SUPPORTED_FILE_EXTENSIONS:
            media_files.extend(input_path.glob(f"*{ext}"))
        
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
        loader_config_from_mapping: Dict[str, Any],
        config: Config,
    ) -> Dict[str, Any]:
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
        """

        output_dir = LoaderHelper.prepare_output_dir(config.paths.markdown_dir)

        result = loader_config_from_mapping.copy()
        result["file_path"] = str(file_path)
        result["output_dir"] = str(output_dir)
        
        return result

