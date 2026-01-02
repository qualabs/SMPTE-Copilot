from __future__ import annotations

"""Factory for creating preprocessor implementations."""

from typing import Any, Callable, ClassVar

from .protocol import Preprocessor
from .rapidfuzz import create_rapidfuzz_preprocessor
from .types import PreprocessorType


class PreprocessorFactory:
    """Factory for creating preprocessor implementations. Easily extensible."""

    _registry: ClassVar[dict[PreprocessorType, Callable[[dict[str, Any]], Preprocessor]]] = {}

    @classmethod
    def register(cls, preprocessor_type: PreprocessorType):
        """Register a new preprocessor factory.

        Parameters
        ----------
        preprocessor_type
            Type to register the preprocessor under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Preprocessor]):
            cls._registry[preprocessor_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, preprocessor_type: PreprocessorType, **kwargs) -> Preprocessor:
        """Create a preprocessor by type.

        Parameters
        ----------
        preprocessor_type
            Type of the preprocessor to create.
        **kwargs
            Additional arguments passed to the preprocessor factory.
        """
        if preprocessor_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown preprocessor: {preprocessor_type}. "
                f"Available preprocessors: {available}"
            )
        return cls._registry[preprocessor_type](kwargs)


PreprocessorFactory.register(PreprocessorType.RAPIDFUZZ)(create_rapidfuzz_preprocessor)

