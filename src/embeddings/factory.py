"""Factory for creating embedding models."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .gemini import create_gemini_embedding
from .huggingface import create_huggingface_embedding
from .openai import create_openai_embedding
from .protocol import Embeddings
from .types import EmbeddingModelType


class EmbeddingModelFactory:
    """Factory for creating embedding models. Easily extensible."""

    _registry: ClassVar[dict[EmbeddingModelType, Callable[[dict[str, Any]], Embeddings]]] = {}

    @classmethod
    def register(cls, model_type: EmbeddingModelType):
        """Register a new embedding model factory.

        Parameters
        ----------
        model_type
            Type to register the model under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Embeddings]):
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
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown model: {model_type}. "
                f"Available models: {available}"
            )
        return cls._registry[model_type](kwargs)

EmbeddingModelFactory.register(EmbeddingModelType.HUGGINGFACE)(create_huggingface_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.OPENAI)(create_openai_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.GEMINI)(create_gemini_embedding)
