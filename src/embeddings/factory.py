"""Factory for creating embedding models."""
from __future__ import annotations

from typing import Dict, Any, Callable

from .protocol import Embeddings
from .types import EmbeddingModelType

from .huggingface import create_huggingface_embedding, create_sentence_transformers_embedding
from .openai import create_openai_embedding

class EmbeddingModelFactory:
    """Factory for creating embedding models. Easily extensible."""
    
    _registry: Dict[EmbeddingModelType, Callable[[Dict[str, Any]], Embeddings]] = {}
    
    @classmethod
    def register(cls, model_type: EmbeddingModelType):
        """Register a new embedding model factory.
        
        Parameters
        ----------
        model_type
            Type to register the model under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Embeddings]):
            cls._registry[model_type] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, model_type: EmbeddingModelType, **kwargs) -> Embeddings:
        """Create an embedding model by type.
        
        Parameters
        ----------
        model_type
            Type of the model to create.
        **kwargs
            Additional arguments passed to the model factory.
            
        Returns
        -------
        Embeddings instance.
        """
        if model_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry.keys())
            raise ValueError(
                f"Unknown model: {model_type}. "
                f"Available models: {available}"
            )
        return cls._registry[model_type](kwargs)

EmbeddingModelFactory.register(EmbeddingModelType.SENTENCE_TRANSFORMERS)(create_sentence_transformers_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.HUGGINGFACE)(create_huggingface_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.OPENAI)(create_openai_embedding)

