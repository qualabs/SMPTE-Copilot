"""Loader types."""
from enum import Enum


class LoaderType(str, Enum):
    """Loader type enumeration."""

    PYMUPDF = "pymupdf"


