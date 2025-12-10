"""Retriever types."""
from enum import Enum


class RetrieverType(str, Enum):
    """Retriever type enumeration."""
    
    SIMILARITY = "similarity"

