"""Gemini (Google Generative AI) embedding model implementation."""
from __future__ import annotations

from typing import Any

from .protocol import Embeddings


def create_gemini_embedding(config: dict[str, Any]) -> Embeddings:
    """Create Gemini embedding model.

    Parameters
    ----------
    config
        Configuration dictionary. Common parameters include:
        - model: str (optional) - Model name (e.g., "models/text-embedding-004")
        - google_api_key: str (optional) - API key (env var GOOGLE_API_KEY also works)
        - Other parameters supported by GoogleGenerativeAIEmbeddings.

    Returns
    -------
    Embeddings instance.

    Raises
    ------
    ValueError
        If model creation fails or invalid parameters are provided.
    ImportError
        If the required dependency is missing.
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:  
        raise ImportError(
            "langchain-google-genai is required for Gemini embeddings. "
            "Install it with `pip install langchain-google-genai`."
        ) from exc

    try:
        return GoogleGenerativeAIEmbeddings(**config)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameter for Gemini embedding model: {exc}. "
            "Check GoogleGenerativeAIEmbeddings documentation for valid parameters."
        ) from exc
    except Exception as exc:  # pragma: no cover - pass-through error path
        raise ValueError(f"Failed to create Gemini embedding model: {exc}") from exc

