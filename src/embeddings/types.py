"""Embedding model types."""
from enum import Enum


class EmbeddingModelType(str, Enum):
    """Embedding model type enumeration."""
    
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"

