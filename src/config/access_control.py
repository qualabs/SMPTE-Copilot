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
    folder_tags_mapping_file: Path | None = Field(
        default=None,
        description="Path to the folder-to-tags mapping JSON file",
    )

    # Query settings
    default_user_role: str | None = Field(
        default=None,
        description="Default user role for query access control (expanded to tags via role_mapping)",
    )
    role_mapping_file: Path | None = Field(
        default=None,
        description="Path to JSON file containing role-to-tags mapping",
    )

    def get_tags_from_file(self, file_path: Path) -> list[str]:
        """Get access tags for a file based on its parent folder name.

        Extracts the immediate parent folder name, looks up tags in the mapping,
        and returns matching tags or falls back to default tags.

        Parameters
        ----------
        file_path : Path
            Path to the file (can be absolute or relative).

        Returns
        -------
        list[str]
            Access tags for the file. Returns tags from mapping if parent folder
            is found, otherwise returns default_tags.
        """
        resolved_path = Path(file_path).resolve()
        logger.info(f"Resolved path: {resolved_path}")
        parent_folder = resolved_path.parent.name
        logger.info(f"Parent folder: {parent_folder}")

        # Handle edge case: file in root directory
        if not parent_folder:
            return self.default_access_tags

        # Load mapping from file
        folder_mapping = {}
        if self.folder_tags_mapping_file is not None:
            try:
                mapping_path = self.folder_tags_mapping_file.expanduser().resolve()
                logger.info(f"Mapping path: {mapping_path}")
                if mapping_path.exists():
                    with mapping_path.open() as f:
                        folder_mapping = json.load(f)
                else:
                    logger.warning(f"Folder tags mapping file not found: {mapping_path}")
            except Exception as e:
                logger.warning(f"Could not load folder tags mapping: {e}")

        # Look up the folder in the mapping
        if parent_folder in folder_mapping:
            logger.info(f"Found tags for folder '{parent_folder}': {folder_mapping[parent_folder]}")
            return folder_mapping[parent_folder]

        # Folder not found in mapping, use default tags
        logger.info(f"Folder '{parent_folder}' not found in mapping, using default tags")
        return self.default_access_tags

