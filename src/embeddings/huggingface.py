"""HuggingFace embedding model implementation."""
from __future__ import annotations

from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings

from .constants import DEFAULT_HUGGINGFACE_MODEL
from .protocol import Embeddings


def create_huggingface_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create HuggingFace embedding model.
    
    Parameters
    ----------
    config
        Configuration dictionary. Common parameters include:
        - model_name: str (optional) - Model name (defaults to DEFAULT_HUGGINGFACE_MODEL)
        - model_kwargs: dict (optional) - Additional model arguments
        - encode_kwargs: dict (optional) - Additional encoding arguments
        - Other parameters supported by HuggingFaceEmbeddings constructor.
        Invalid parameters will be caught by HuggingFaceEmbeddings and raise clear errors.
    
    Returns
    -------
    Embeddings instance.
    
    Raises
    ------
    ValueError
        If model creation fails or invalid parameters are provided.
    """
    
    # Use default model_name if not provided
    if "model_name" not in config:
        config = {**config, "model_name": DEFAULT_HUGGINGFACE_MODEL}
    
    try:
        return HuggingFaceEmbeddings(**config)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameter for HuggingFace embedding model: {e}. "
            "Check HuggingFaceEmbeddings documentation for valid parameters."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to create HuggingFace embedding model: {e}") from e