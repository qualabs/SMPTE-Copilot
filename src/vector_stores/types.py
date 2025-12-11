"""Vector store types."""
from enum import Enum


class VectorStoreType(str, Enum):
    """Vector store type enumeration."""

    CHROMADB = "chromadb"


