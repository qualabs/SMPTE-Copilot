"""Factory for creating tokenizer implementations."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .gemini import create_gemini_tokenizer
from .protocol import Tokenizer
from .simple import create_simple_tokenizer
from .types import TokenizerType


class TokenizerFactory:
    """Factory for creating tokenizer implementations. Easily extensible."""

    _registry: ClassVar[dict[TokenizerType, Callable[[dict[str, Any]], Tokenizer]]] = {}

    @classmethod
    def register(cls, tokenizer_type: TokenizerType):
        """Register a new tokenizer factory.

        Parameters
        ----------
        tokenizer_type
            Type to register the tokenizer under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Tokenizer]):
            cls._registry[tokenizer_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, tokenizer_type: TokenizerType, **kwargs) -> Tokenizer:
        """Create a tokenizer by type.

        Parameters
        ----------
        tokenizer_type
            Type of the tokenizer to create.
        **kwargs
            Additional arguments passed to the tokenizer factory.

        Returns
        -------
        Tokenizer instance.
        """
        if tokenizer_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown tokenizer: {tokenizer_type}. "
                f"Available tokenizers: {available}"
            )
        return cls._registry[tokenizer_type](kwargs)


TokenizerFactory.register(TokenizerType.SIMPLE)(create_simple_tokenizer)
TokenizerFactory.register(TokenizerType.GEMINI)(create_gemini_tokenizer)

