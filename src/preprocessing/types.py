"""Preprocessor types."""

from enum import Enum


class PreprocessorType(str, Enum):
    """Preprocessor type enumeration."""

    RAPIDFUZZ = "rapidfuzz"

