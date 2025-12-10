"""OpenAI embedding model implementation."""
from __future__ import annotations

from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings

from .protocol import Embeddings


def create_openai_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create OpenAI embedding model."""
    return OpenAIEmbeddings(**{k: v for k, v in config.items() if k != "model_name"})

