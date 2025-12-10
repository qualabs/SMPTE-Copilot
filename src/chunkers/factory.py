"""Factory for creating chunker implementations."""
from __future__ import annotations

from typing import Dict, Any, Callable

from .protocol import Chunker
from .langchain_chunker import create_langchain_chunker


class ChunkerFactory:
    """Factory for creating chunker implementations. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], Chunker]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new chunker factory.
        
        Parameters
        ----------
        name
            Name to register the chunker under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Chunker]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, chunker_name: str, **kwargs) -> Chunker:
        """Create a chunker by name.
        
        Parameters
        ----------
        chunker_name
            Name of the chunker to create.
        **kwargs
            Additional arguments passed to the chunker factory.
            
        Returns
        -------
        Chunker instance.
        """
        if chunker_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown chunker: {chunker_name}. "
                f"Available chunkers: {available}"
            )
        return cls._registry[chunker_name](kwargs)

# Register default chunkers
ChunkerFactory.register("langchain")(create_langchain_chunker)
