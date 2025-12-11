"""Protocol for embedding model implementations."""
from __future__ import annotations

from typing import Protocol


class Embeddings(Protocol):
    """Protocol for embedding model implementations.

    Compatible with LangChain's Embeddings interface and any custom
    implementation that provides these methods.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

