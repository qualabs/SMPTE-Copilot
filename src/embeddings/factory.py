"""Factory for creating embedding models."""
from __future__ import annotations

from typing import List, Dict, Any, Callable

from langchain.embeddings.base import Embeddings as LangChainEmbeddings

from .protocol import Embeddings

from .huggingface import create_huggingface_embedding, create_sentence_transformers_embedding
from .openai import create_openai_embedding

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

EmbeddingModelFactory.register("sentence-transformers")(create_sentence_transformers_embedding)
EmbeddingModelFactory.register("huggingface")(create_huggingface_embedding)
EmbeddingModelFactory.register("openai")(create_openai_embedding)

