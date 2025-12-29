"""Gemini (Google Generative AI) embedding model implementation."""
from __future__ import annotations

from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .protocol import Embeddings


def create_gemini_embedding(config: dict[str, Any]) -> Embeddings:
    """Create Gemini embedding model.

    Parameters
    ----------
    config
        Configuration dictionary. Common parameters include:
        - model: str (optional) - Model name (e.g., "models/text-embedding-004")
        - google_api_key: str (optional)
        - Other parameters supported by GoogleGenerativeAIEmbeddings.
        Invalid parameters will be caught by GoogleGenerativeAIEmbeddings and raise clear errors.
    """
    try:
        return GoogleGenerativeAIEmbeddings(**config)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameter for Gemini embedding model: {e}. "
            "Check GoogleGenerativeAIEmbeddings documentation for valid parameters."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to create Gemini embedding model: {e}") from e

