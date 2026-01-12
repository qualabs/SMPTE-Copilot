from __future__ import annotations

"""Factory for creating reranker backends."""

from typing import Any, Callable, ClassVar

from .gemini import create_gemini_reranker
from .protocol import Reranker
from .types import RerankerType


class RerankerFactory:
    """Factory for creating reranker backends. Easily extensible."""

    _registry: ClassVar[dict[RerankerType, Callable[[dict[str, Any]], Reranker]]] = {}

    @classmethod
    def register(cls, reranker_type: RerankerType):
        """Register a new reranker factory.

        Parameters
        ----------
        reranker_type
            Type to register the reranker under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Reranker]):
            cls._registry[reranker_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, reranker_type: RerankerType, **kwargs) -> Reranker:
        """Create a reranker by type.

        Parameters
        ----------
        reranker_type
            Type of the reranker to create.
        **kwargs
            Additional arguments passed to the reranker factory.

        Returns
        -------
        Reranker instance.
        """
        if reranker_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown reranker: {reranker_type}. "
                f"Available rerankers: {available}"
            )
        return cls._registry[reranker_type](kwargs)


RerankerFactory.register(RerankerType.GEMINI)(create_gemini_reranker)
