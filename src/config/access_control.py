"""Access control configuration."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AccessControlConfig(BaseSettings):
    """Access control configuration for document ingestion and querying."""

    # Ingestion settings
    default_access_tags: list[str] = Field(
        default_factory=list,
        description="Default access tags for ingested documents (comma-separated or list)",
    )
    default_required_role: Optional[str] = Field(
        default=None,
        description="Default required role for strict access control on ingested documents",
    )

    # Query settings
    default_user_role: Optional[str] = Field(
        default=None,
        description="Default user role for query access control",
    )
    default_user_tags: list[str] = Field(
        default_factory=list,
        description="Default user tags for query access control (comma-separated or list)",
    )
    role_mapping_file: Optional[Path] = Field(
        default=None,
        description="Path to JSON file containing role-to-tags mapping",
    )

