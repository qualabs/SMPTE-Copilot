"""Vector store utilities for RAG ingestion."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from pathlib import Path

from langchain.schema import Document


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
    def create(cls, store_name: str, **kwargs) -> Any:
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
@VectorStoreFactory.register("chromadb")
def _create_chromadb(config: Dict[str, Any]) -> Any:
    """Create ChromaDB vector store."""
    try:
        from langchain_community.vectorstores import Chroma
        import chromadb
    except ImportError:
        raise ImportError(
            "ChromaDB requires 'chromadb' and 'langchain-community' packages. "
            "Install with: pip install chromadb langchain-community"
        )
    
    # Get configuration
    persist_directory = config.get("persist_directory", "./chroma_db")
    collection_name = config.get("collection_name", "rag_documents")
    embedding_function = config.get("embedding_function")
    
    if embedding_function is None:
        raise ValueError(
            "ChromaDB requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )
    
    # Create ChromaDB instance
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


@VectorStoreFactory.register("qdrant")
def _create_qdrant(config: Dict[str, Any]) -> Any:
    """Create Qdrant vector store."""
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError(
            "Qdrant requires 'qdrant-client' and 'langchain-qdrant' packages. "
            "Install with: pip install qdrant-client langchain-qdrant"
        )
    
    # Get configuration
    url = config.get("url", "http://localhost:6333")
    collection_name = config.get("collection_name", "rag_documents")
    embedding_function = config.get("embedding_function")
    
    if embedding_function is None:
        raise ValueError(
            "Qdrant requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )
    
    # Create Qdrant client
    client = QdrantClient(url=url)
    
    # Check if collection exists, if not create it manually
    if not client.collection_exists(collection_name):
        from qdrant_client.models import Distance, VectorParams
        
        # Get embedding dimension from the embedding function
        # Create a test embedding to determine the dimension
        test_embedding = embedding_function.embed_query("test")
        embedding_dim = len(test_embedding)
        
        # Create collection with proper vector configuration
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    
    # Create Qdrant vector store using the new langchain-qdrant package
    # Pass the client directly
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
    )


class VectorStoreIngester:
    """Ingest documents with embeddings into vector stores. Store-agnostic interface."""

    def __init__(
        self,
        vector_store: Any = None,
        store_name: str = "chromadb",
        store_config: Optional[Dict[str, Any]] = None,
        embedding_function: Any = None,
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
            For ChromaDB: {"persist_directory": "./chroma_db", "collection_name": "docs"}
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
        
        if has_embeddings and self.store_name == "chromadb":
            # For ChromaDB, extract embeddings and texts separately
            texts = [chunk.page_content for chunk in chunks]
            embeddings = [chunk.metadata.get("embedding") for chunk in chunks]
            
            # ChromaDB doesn't support list values in metadata
            # Convert access_tags list to comma-separated string
            metadatas = []
            for chunk in chunks:
                metadata = {k: v for k, v in chunk.metadata.items() if k != "embedding"}
                
                # Convert access_tags list to string if present
                if "access_tags" in metadata and isinstance(metadata["access_tags"], list):
                    metadata["access_tags"] = ",".join(metadata["access_tags"])
                
                metadatas.append(metadata)
            
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Use add_texts with embeddings for ChromaDB
            self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        elif has_embeddings and self.store_name == "qdrant":
            # For Qdrant, keep native list support for access_tags
            # Remove embeddings from metadata since Qdrant will compute them
            cleaned_chunks = []
            for chunk in chunks:
                metadata = {k: v for k, v in chunk.metadata.items() if k != "embedding"}
                cleaned_chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
            
            # Use add_documents - Qdrant will compute embeddings using the embedding function
            self.vector_store.add_documents(cleaned_chunks)
        else:
            # Let the vector store compute embeddings automatically
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

    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
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
        if self.store_name == "qdrant":
            # Qdrant persists automatically (client-server)
            pass
        elif hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        elif hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "persist"):
            # For ChromaDB
            self.vector_store._collection.persist()

    def delete_collection(self) -> None:
        """Delete the collection (if supported)."""
        if self.store_name == "qdrant":
            # For Qdrant, use client to delete collection
            try:
                if hasattr(self.vector_store, "client"):
                    collection_name = self.vector_store.collection_name
                    self.vector_store.client.delete_collection(collection_name)
            except Exception:
                pass
        elif hasattr(self.vector_store, "delete"):
            self.vector_store.delete()
        elif hasattr(self.vector_store, "_collection"):
            # For ChromaDB
            try:
                self.vector_store._collection.delete()
            except Exception:
                pass


__all__ = ["VectorStoreIngester"]
