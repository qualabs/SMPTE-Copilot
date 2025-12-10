"""HuggingFace embedding model implementation."""
from __future__ import annotations

import inspect
from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings

from .constants import DEFAULT_HUGGINGFACE_MODEL
from .protocol import Embeddings


def create_huggingface_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create HuggingFace embedding model."""
    # Get valid parameters from HuggingFaceEmbeddings constructor
    sig = inspect.signature(HuggingFaceEmbeddings.__init__)
    valid_params = set(sig.parameters.keys())
    
    # Filter config to only include valid parameters
    filtered_config = {
        k: v for k, v in config.items()
        if k in valid_params
    }
    
    # Use default model_name if not provided
    if "model_name" not in filtered_config:
        filtered_config["model_name"] = DEFAULT_HUGGINGFACE_MODEL
    
    return HuggingFaceEmbeddings(**filtered_config)