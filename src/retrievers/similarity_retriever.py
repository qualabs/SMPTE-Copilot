"""Similarity search retriever implementation."""
from __future__ import annotations

from typing import List, Dict, Any

from langchain.schema import Document

from ..constants import DEFAULT_RETRIEVAL_K
from .protocol import Retriever
from ..vector_stores.protocol import VectorStore


class DocumentRetriever:
    """Retrieve relevant documents from vector store using similarity search.
    
    This is a concrete implementation of the Retriever protocol using
    similarity search on a vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = DEFAULT_RETRIEVAL_K,
    ):
        """Initialize the retriever.

        Parameters
        ----------
        vector_store
            Vector store instance (created via VectorStoreFactory).
        k
            Number of documents to retrieve. Default: DEFAULT_RETRIEVAL_K
        """
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.

        Parameters
        ----------
        query
            The search query.

        Returns
        -------
        List of Document objects, most relevant first.
        """
        return self.vector_store.similarity_search(query, k=self.k)

    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve documents with similarity scores.

        Parameters
        ----------
        query
            The search query.

        Returns
        -------
        List of tuples: (Document, score), most relevant first.
        """
        return self.vector_store.similarity_search_with_score(query, k=self.k)


def create_similarity_retriever(config: Dict[str, Any]) -> Retriever:
    vector_store = config.get("vector_store")
    if vector_store is None:
        raise ValueError("vector_store is required for similarity retriever")
    k = config.get("k")
    return DocumentRetriever(vector_store, k=k)

