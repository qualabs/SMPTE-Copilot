"""Embedding factory and chunk embedder implementations."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable

from langchain.schema import Document
from langchain.embeddings.base import Embeddings as LangChainEmbeddings

from .protocol import Embeddings


class EmbeddingModelFactory:
    """Factory for creating embedding models. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], LangChainEmbeddings]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new embedding model factory.
        
        Parameters
        ----------
        name
            Name to register the model under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], LangChainEmbeddings]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> Embeddings:
        """Create an embedding model by name.
        
        Parameters
        ----------
        model_name
            Name of the model to create.
        **kwargs
            Additional arguments passed to the model factory.
            
        Returns
        -------
        Embeddings instance.
        """
        if model_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available}"
            )
        return cls._registry[model_name](kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._registry.keys())


# Register default embedding models
from .huggingface import create_huggingface_embedding, create_sentence_transformers_embedding
from .openai import create_openai_embedding

EmbeddingModelFactory.register("sentence-transformers")(create_sentence_transformers_embedding)
EmbeddingModelFactory.register("huggingface")(create_huggingface_embedding)
EmbeddingModelFactory.register("openai")(create_openai_embedding)


class ChunkEmbedder:
    """Embed chunks for RAG vector search. Model-agnostic interface."""

    def __init__(
        self,
        embedding_model: Optional[Embeddings] = None,
        model_name: str = "huggingface",
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the embedder.

        Parameters
        ----------
        embedding_model
            Pre-initialized LangChain Embeddings object.
            If provided, model_name and model_config are ignored.
        model_name
            Name of the embedding model to use.
            Use EmbeddingModelFactory.list_models() to see available models.
            Default: "huggingface" (free, no API key needed, runs locally)
        model_config
            Optional configuration dictionary passed to the model factory.
            For example: {"model_name": "custom-model-name"} for sentence-transformers.
        """
        if embedding_model is not None:
            # Use provided model directly - completely independent
            self.embedding_model = embedding_model
            self.model_name = "custom"
        else:
            # Create model using factory - easy to swap
            config = model_config or {}
            self.embedding_model = EmbeddingModelFactory.create(model_name, **config)
            self.model_name = model_name

    def embed_chunks(self, chunks: List[Document]) -> List[Document]:
        """Embed a list of document chunks.

        Parameters
        ----------
        chunks
            List of LangChain Document objects to embed.

        Returns
        -------
        List of Document objects with embeddings stored in metadata.
        The interface is consistent regardless of the underlying model.
        """
        if not chunks:
            return []

        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings - model-agnostic interface
        embeddings = self.embedding_model.embed_documents(texts)

        # Add embeddings to document metadata
        # This format is consistent regardless of model
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create a copy to avoid modifying original
            embedded_chunk = Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "embedding": embedding,
                    "embedding_model": self.model_name,  # Track which model was used
                }
            )
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        Parameters
        ----------
        query
            The search query text to embed.

        Returns
        -------
        Embedding vector as a list of floats.
        Consistent format regardless of model.
        """
        return self.embedding_model.embed_query(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns
        -------
        Dimension of the embedding vectors.
        """
        # Test with a small text to get dimension
        test_embedding = self.embedding_model.embed_query("test")
        return len(test_embedding)

    def swap_model(
        self,
        model_name: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Swap to a different embedding model without affecting the interface.

        Parameters
        ----------
        model_name
            Name of the new embedding model to use.
        model_config
            Optional configuration for the new model.
        """
        config = model_config or {}
        self.embedding_model = EmbeddingModelFactory.create(model_name, **config)
        self.model_name = model_name

