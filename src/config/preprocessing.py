"""Preprocessing configuration."""

from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.preprocessing.types import PreprocessorType


class PreprocessingConfig(BaseSettings):
    """Preprocessing configuration for removing repeated content."""

    preprocessing_name: PreprocessorType = Field(
        default=PreprocessorType.RAPIDFUZZ,
        description="Preprocessor type",
    )
    preprocessing_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Preprocessor-specific configuration dictionary. "
        "For rapidfuzz: min_repetitions, similarity_threshold.",
    )

