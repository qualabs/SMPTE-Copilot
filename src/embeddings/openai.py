"""OpenAI embedding model implementation."""
from __future__ import annotations

from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings

from .protocol import Embeddings


def create_openai_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create OpenAI embedding model.
    
    Parameters
    ----------
    config
        Configuration dictionary. Common parameters include:
        - model: str (optional) - Model name, e.g., "text-embedding-3-small"
        - openai_api_key: str (optional) - API key (can also use OPENAI_API_KEY env var)
        - Other parameters supported by OpenAIEmbeddings constructor.
        Invalid parameters will be caught by OpenAIEmbeddings and raise clear errors.
    
    Returns
    -------
    Embeddings instance.
    
    Raises
    ------
    ValueError
        If model creation fails or invalid parameters are provided.
    """
    try:
        return OpenAIEmbeddings(**config)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameter for OpenAI embedding model: {e}. "
            "Check OpenAIEmbeddings documentation for valid parameters."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI embedding model: {e}") from e

