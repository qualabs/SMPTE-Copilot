"""Retriever factory for creating retriever implementations."""
from __future__ import annotations

from typing import Dict, Any, Callable

from .protocol import Retriever
from .similarity_retriever import create_similarity_retriever


class RetrieverFactory:
    """Factory for creating retriever implementations. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], Retriever]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new retriever factory.
        
        Parameters
        ----------
        name
            Name to register the retriever under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Retriever]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, retriever_name: str, **kwargs) -> Retriever:
        """Create a retriever by name.
        
        Parameters
        ----------
        retriever_name
            Name of the retriever to create.
        **kwargs
            Additional arguments passed to the retriever factory.
            
        Returns
        -------
        Retriever instance.
        """
        if retriever_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown retriever: {retriever_name}. "
                f"Available retrievers: {available}"
            )
        return cls._registry[retriever_name](kwargs)


# Register default retrievers
RetrieverFactory.register("similarity")(create_similarity_retriever)