"""OpenAI embedding model implementation."""
from __future__ import annotations

import inspect
from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings

from .protocol import Embeddings


def create_openai_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create OpenAI embedding model."""
    # Get valid parameters from OpenAIEmbeddings constructor
    sig = inspect.signature(OpenAIEmbeddings.__init__)
    valid_params = set(sig.parameters.keys())
    
    # Filter config to only include valid parameters
    filtered_config = {
        k: v for k, v in config.items()
        if k in valid_params
    }
    
    try:
        return OpenAIEmbeddings(**filtered_config)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI embedding model: {e}") from e

