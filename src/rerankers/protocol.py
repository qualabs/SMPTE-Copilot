from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Reranker(Protocol):
    """Protocol for document reranking backends."""

    def rerank(
        self, query: str, documents: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Rerank documents based on relevance to the query.

        Parameters
        ----------
        query
            User's search query
        documents
            List of (Document, score) tuples from initial retrieval

        Returns
        -------
        List of (Document, score) tuples reordered by relevance with new scores
        """
        ...
