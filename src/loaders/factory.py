"""Factory for creating document loader implementations."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .protocol import DocumentLoader
from .pymupdf_loader import create_pymupdf_loader
from .types import LoaderType


class LoaderFactory:
    """Factory for creating document loader implementations. Easily extensible."""

    _registry: ClassVar[dict[LoaderType, Callable[[dict[str, Any]], DocumentLoader]]] = {}

    @classmethod
    def register(cls, loader_type: LoaderType):
        """Register a new loader factory.

        Parameters
        ----------
        loader_type
            Type to register the loader under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], DocumentLoader]):
            cls._registry[loader_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, loader_type: LoaderType, **kwargs) -> DocumentLoader:
        """Create a loader by type.

        Parameters
        ----------
        loader_type
            Type of the loader to create.
        **kwargs
            Additional arguments passed to the loader factory.

        Returns
        -------
        DocumentLoader instance.
        """
        if loader_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown loader: {loader_type}. "
                f"Available loaders: {available}"
            )
        return cls._registry[loader_type](kwargs)


# Register default loaders
LoaderFactory.register(LoaderType.PYMUPDF)(create_pymupdf_loader)

