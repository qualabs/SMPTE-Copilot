"""Access control configuration."""

import json
import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class AccessControlConfig(BaseSettings):
    """Access control configuration for document ingestion and querying."""

    # Ingestion settings
    default_access_tags: list[str] = Field(
        default_factory=list,
        description="Default access tags for ingested documents (comma-separated or list)",
    )
    default_required_role: str | None = Field(
        default=None,
        description="Default required role for strict access control on ingested documents",
    )

    # Query settings
    default_user_role: str | None = Field(
        default=None,
        description="Default user role for query access control (expanded to tags via role_mapping)",
    )
    notify_on_denied_access: bool = Field(
        default=False,
        description="If true, notify users about restricted documents instead of silent filtering. "
        "When false (default), uses efficient Qdrant filtering. When true, retrieves all documents "
        "and separates accessible vs restricted, showing restricted document sources in the response.",
    )

    # Unified access mapping file
    access_mapping_file: Path | None = Field(
        default=None,
        description="Path to JSON file containing unified folder-to-tags and role-to-tags mapping",
    )

    # Internal cached mappings (not part of config)
    _folder_mapping: dict[str, list[str]] | None = None
    _role_mapping: dict[str, list[str]] | None = None

    def _load_access_mapping(self) -> None:
        """Load the unified access mapping file containing folders and roles.

        This method loads the file once and caches both mappings internally.
        """
        if self._folder_mapping is not None and self._role_mapping is not None:
            return  # Already loaded

        if self.access_mapping_file is None:
            logger.info("No access_mapping_file configured")
            self._folder_mapping = {}
            self._role_mapping = {}
            return

        try:
            mapping_path = self.access_mapping_file.expanduser().resolve()
            logger.info(f"Loading access mapping from: {mapping_path}")

            if not mapping_path.exists():
                logger.warning(f"Access mapping file not found: {mapping_path}")
                self._folder_mapping = {}
                self._role_mapping = {}
                return

            with mapping_path.open() as f:
                data = json.load(f)

            # Validate structure
            if not isinstance(data, dict):
                logger.error(f"Invalid access mapping format: expected dict, got {type(data)}")
                self._folder_mapping = {}
                self._role_mapping = {}
                return

            self._folder_mapping = data.get("folders", {})
            self._role_mapping = data.get("roles", {})

            logger.info(f"Loaded {len(self._folder_mapping)} folder mappings and {len(self._role_mapping)} role mappings")

        except Exception as e:
            logger.info(f"Could not load access mapping: {e}")
            self._folder_mapping = {}
            self._role_mapping = {}

    def get_role_mapping(self) -> dict[str, list[str]]:
        """Get the role-to-tags mapping.

        Returns
        -------
        dict[str, list[str]]
            Role-to-tags mapping. Empty dict if file doesn't exist or can't be loaded.
        """
        self._load_access_mapping()
        return self._role_mapping or {}

    def get_tags_from_file(self, file_path: Path | str) -> list[str]:
        """Get access tags for a file based on its parent folder name.

        Extracts the immediate parent folder name, looks up tags in the mapping,
        and returns matching tags or falls back to default tags.

        Parameters
        ----------
        file_path : Path | str
            Path to the file (can be absolute, relative, or URI).
            The parent folder is extracted using generic path parsing.

        Returns
        -------
        list[str]
            Access tags for the file. Returns tags from mapping if parent folder
            is found, otherwise returns default_tags.
        """
        # Use Path for generic path/URI parsing without resolving to filesystem
        # This works for both local paths and URIs (s3://, http://, etc.)
        path_obj = Path(str(file_path))
        parent_folder = path_obj.parent.name

        logger.info(f"File path: {file_path}")
        logger.info(f"Parent folder: {parent_folder}")

        # Handle edge case: file in root directory
        if not parent_folder:
            return self.default_access_tags

        # Load mapping (cached after first call)
        self._load_access_mapping()
        folder_mapping = self._folder_mapping or {}

        # Look up the folder in the mapping
        if parent_folder in folder_mapping:
            logger.info(f"Found tags for folder '{parent_folder}': {folder_mapping[parent_folder]}")
            return folder_mapping[parent_folder]

        # Folder not found in mapping, use default tags
        logger.info(f"Folder '{parent_folder}' not found in mapping, using default tags")
        return self.default_access_tags

