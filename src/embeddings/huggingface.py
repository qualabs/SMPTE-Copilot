"""HuggingFace embedding model implementation."""
from __future__ import annotations

from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings

from .constants import DEFAULT_HUGGINGFACE_MODEL
from .protocol import Embeddings


def create_huggingface_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create HuggingFace embedding model."""
    model_name = config.get(
        "model_name",
        DEFAULT_HUGGINGFACE_MODEL
    )
    
    # Filter out model_name from config before passing to constructor
    filtered_config = {k: v for k, v in config.items() if k != "model_name"}
    
    # Explicitly pass model_name to avoid deprecation warning
    return HuggingFaceEmbeddings(model_name=model_name, **filtered_config)


def create_sentence_transformers_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create sentence-transformers embedding model."""
    model_name = config.get(
        "model_name", 
        DEFAULT_HUGGINGFACE_MODEL
    )
    # Explicitly pass model_name to avoid deprecation warning
    return HuggingFaceEmbeddings(model_name=model_name)

