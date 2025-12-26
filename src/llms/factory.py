"""Factory for creating LLM backends."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from src.llms.gemini import create_gemini_llm

from .protocol import LLM
from .types import LLMType


class LLMFactory:
    """Factory for creating LLM backends. Easily extensible."""

    _registry: ClassVar[dict[LLMType, Callable[[dict[str, Any]], LLM]]] = {}

    @classmethod
    def register(cls, llm_type: LLMType):
        """Register a new LLM factory.

        Parameters
        ----------
        llm_type
            Type to register the LLM under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], LLM]):
            cls._registry[llm_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, llm_type: LLMType, **kwargs) -> LLM:
        """Create an LLM by type.

        Parameters
        ----------
        llm_type
            Type of the LLM to create.
        **kwargs
            Additional arguments passed to the LLM factory.

        Returns
        -------
        LLM instance.
        """
        if llm_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown LLM: {llm_type}. "
                f"Available LLMs: {available}"
            )
        return cls._registry[llm_type](kwargs)


LLMFactory.register(LLMType.GEMINI)(create_gemini_llm)
