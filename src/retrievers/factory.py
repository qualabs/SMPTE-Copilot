"""Retriever factory for creating retriever implementations."""
from __future__ import annotations

from typing import Dict, Any, Callable

from .protocol import Retriever
from .types import RetrieverType
from .similarity_retriever import create_similarity_retriever


class RetrieverFactory:
    """Factory for creating retriever implementations. Easily extensible."""
    
    _registry: Dict[RetrieverType, Callable[[Dict[str, Any]], Retriever]] = {}
    
    @classmethod
    def register(cls, retriever_type: RetrieverType):
        """Register a new retriever factory.
        
        Parameters
        ----------
        retriever_type
            Type to register the retriever under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Retriever]):
            cls._registry[retriever_type] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, retriever_type: RetrieverType, **kwargs) -> Retriever:
        """Create a retriever by type.
        
        Parameters
        ----------
        retriever_type
            Type of the retriever to create.
        **kwargs
            Additional arguments passed to the retriever factory.
            
        Returns
        -------
        Retriever instance.
        """
        if retriever_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry.keys())
            raise ValueError(
                f"Unknown retriever: {retriever_type}. "
                f"Available retrievers: {available}"
            )
        return cls._registry[retriever_type](kwargs)


# Register default retrievers
RetrieverFactory.register(RetrieverType.SIMILARITY)(create_similarity_retriever)