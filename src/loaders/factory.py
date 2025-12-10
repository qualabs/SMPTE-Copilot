"""Factory for creating document loader implementations."""
from __future__ import annotations

from typing import Dict, Any, Callable

from .protocol import DocumentLoader
from .pymupdf_loader import create_pymupdf_loader


class LoaderFactory:
    """Factory for creating document loader implementations. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], DocumentLoader]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new loader factory.
        
        Parameters
        ----------
        name
            Name to register the loader under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], DocumentLoader]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, loader_name: str, **kwargs) -> DocumentLoader:
        """Create a loader by name.
        
        Parameters
        ----------
        loader_name
            Name of the loader to create.
        **kwargs
            Additional arguments passed to the loader factory.
            
        Returns
        -------
        DocumentLoader instance.
        """
        if loader_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown loader: {loader_name}. "
                f"Available loaders: {available}"
            )
        return cls._registry[loader_name](kwargs)


# Register default loaders
LoaderFactory.register("pymupdf")(create_pymupdf_loader)

