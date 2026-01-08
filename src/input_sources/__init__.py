"""Input sources for accessing files from different locations."""

from .factory import InputSourceFactory
from .protocol import InputSource
from .types import InputSourceType

__all__ = [
    "InputSource",
    "InputSourceFactory",
    "InputSourceType",
]
