from __future__ import annotations

"""ChromaDB vector store implementation."""

from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from ..constants import DEFAULT_RETRIEVAL_K
from ..embeddings.protocol import Embeddings
from .constants import CHUNK_ID_PREFIX, DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_DB_DIR
from .protocol import VectorStore


class ChromaDBWrapper:
    """Wrapper for ChromaDB to convert int IDs to strings."""

    def __init__(self, chroma_store: Chroma):
        """Initialize the wrapper.

        Parameters
        ----------
        chroma_store
            The underlying Chroma instance.
        """
        self._store = chroma_store

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[int]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Add texts to the vector store with pre-computed embeddings.

        Parameters
        ----------
        texts
            List of text strings to add.
        metadatas
            Optional list of metadata dictionaries.
        ids
            Optional list of document IDs (integers).
        embeddings
            Pre-computed embedding vectors.
        """
        # Convert int IDs to strings with prefix for ChromaDB
        string_ids = [f"{CHUNK_ID_PREFIX}{doc_id}" for doc_id in ids] if ids else None

        self._store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=string_ids,
            embeddings=embeddings,
        )

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[Document]:
        """Search for similar documents.
        
        Note: ChromaDB does not support metadata filtering in this implementation.
        The filter parameter is ignored.
        """
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Note: ChromaDB does not support metadata filtering in this implementation.
        The filter parameter is ignored.
        """
        return self._store.similarity_search_with_score(query, k=k)

    def add_documents(self, documents: list[Document]) -> list[int]:
        """Add documents to the vector store."""
        result_ids = self._store.add_documents(documents)
        # ChromaDB returns string IDs, try to convert back to ints
        converted_ids = []
        for doc_id in result_ids:
            converted_ids.append(int(doc_id.replace(CHUNK_ID_PREFIX, "")))
        return converted_ids

    def persist(self) -> None:
        """Persist the vector store to disk."""
        self._store.persist()


def create_chromadb_store(config: dict[str, Any]) -> VectorStore:
    """Create a ChromaDB vector store from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - embedding_function: Embeddings (required) - Embedding model instance
        - persist_directory: str (optional) - Directory to persist the database
        - collection_name: str (optional) - Name of the collection

    Returns
    -------
    VectorStore instance.

    Raises
    ------
    ValueError
        If embedding_function is not provided.
    """
    persist_directory = config.get("persist_directory", DEFAULT_VECTOR_DB_DIR)
    collection_name = config.get("collection_name", DEFAULT_COLLECTION_NAME)
    embedding_function: Embeddings = config.get("embedding_function")

    if embedding_function is None:
        raise ValueError(
            "ChromaDB requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )

    vector_db_path = Path(persist_directory).expanduser().resolve()
    if not vector_db_path.exists():
        raise RuntimeError(
            f"Vector database not found at {vector_db_path}. "
            "Please run ingestion first."
        )

    chroma_store = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return ChromaDBWrapper(chroma_store)

