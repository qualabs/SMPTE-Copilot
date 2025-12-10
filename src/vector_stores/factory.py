"""Vector store factory and ingester implementations."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable

from langchain.schema import Document

from .protocol import VectorStore
from ..embeddings.protocol import Embeddings


class VectorStoreFactory:
    """Factory for creating vector stores. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new vector store factory.
        
        Parameters
        ----------
        name
            Name to register the vector store under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Any]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, store_name: str, **kwargs) -> VectorStore:
        """Create a vector store by name.
        
        Parameters
        ----------
        store_name
            Name of the vector store to create.
        **kwargs
            Additional arguments passed to the store factory.
            
        Returns
        -------
        Vector store instance.
        """
        if store_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown vector store: {store_name}. "
                f"Available stores: {available}"
            )
        return cls._registry[store_name](kwargs)
    
    @classmethod
    def list_stores(cls) -> List[str]:
        """List all registered vector store names."""
        return list(cls._registry.keys())


# Register vector store implementations
from .chromadb import create_chromadb_store

VectorStoreFactory.register("chromadb")(create_chromadb_store)


class VectorStoreIngester:
    """Ingest documents with embeddings into vector stores. Store-agnostic interface."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        store_name: str = "chromadb",
        store_config: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Embeddings] = None,
    ):
        """Initialize the vector store ingester.

        Parameters
        ----------
        vector_store
            Pre-initialized vector store object.
            If provided, store_name and store_config are ignored.
        store_name
            Name of the vector store to use.
            Use VectorStoreFactory.list_stores() to see available stores.
            Default: "chromadb"
        store_config
            Optional configuration dictionary passed to the store factory.
            For ChromaDB: {"persist_directory": "./vector_db", "collection_name": "docs"}
        embedding_function
            Embedding function/model to use for the vector store.
            Required if creating a new store.
        """
        if vector_store is not None:
            # Use provided store directly - completely independent
            self.vector_store = vector_store
            self.store_name = "custom"
        else:
            # Create store using factory - easy to swap
            config = store_config or {}
            if embedding_function is None:
                raise ValueError(
                    "embedding_function is required when creating a new vector store. "
                    "Pass the embedding model from ChunkEmbedder.embedding_model"
                )
            config["embedding_function"] = embedding_function
            self.vector_store = VectorStoreFactory.create(store_name, **config)
            self.store_name = store_name

    def ingest_chunks(self, chunks: List[Document]) -> None:
        """Ingest document chunks with embeddings into the vector store.

        Parameters
        ----------
        chunks
            List of Document objects with embeddings in metadata.
            Embeddings should be in chunk.metadata['embedding'].
            If embeddings are present, they will be used; otherwise,
            the vector store will compute them using the embedding function.
        """
        if not chunks:
            return

        # Check if chunks have pre-computed embeddings
        has_embeddings = any("embedding" in chunk.metadata for chunk in chunks)
        
        # Try to use add_texts with pre-computed embeddings if available
        # This is more efficient than letting the store recompute embeddings
        if has_embeddings and hasattr(self.vector_store, "add_texts"):
            try:
                # Extract embeddings and texts separately
                texts = [chunk.page_content for chunk in chunks]
                embeddings = [chunk.metadata.get("embedding") for chunk in chunks]
                metadatas = [
                    {k: v for k, v in chunk.metadata.items() if k != "embedding"}
                    for chunk in chunks
                ]
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                
                # Use add_texts with embeddings (if supported by the store)
                self.vector_store.add_texts(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                return
            except (TypeError, AttributeError):
                # Store doesn't support add_texts with embeddings, fall back
                pass
        
        # Fallback: Let the vector store compute embeddings automatically
        self.vector_store.add_documents(chunks)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search the vector store for similar documents.

        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.

        Returns
        -------
        List of Document objects, most similar first.
        """
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """Search the vector store with similarity scores.

        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.

        Returns
        -------
        List of tuples: (Document, score), most similar first.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def persist(self) -> None:
        """Persist the vector store to disk (if supported)."""
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        elif hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "persist"):
            # For ChromaDB
            self.vector_store._collection.persist()

    def delete_collection(self) -> None:
        """Delete the collection (if supported)."""
        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete()
        elif hasattr(self.vector_store, "_collection"):
            # For ChromaDB
            try:
                self.vector_store._collection.delete()
            except Exception:
                pass

