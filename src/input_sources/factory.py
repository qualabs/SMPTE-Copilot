from __future__ import annotations

"""Input source factory with dynamic registration."""

from collections.abc import Callable
from typing import Any, ClassVar

from .local_source import create_local_source
from .protocol import InputSource
from .s3_source import create_s3_source
from .types import InputSourceType


class InputSourceFactory:
    """Factory for creating input sources with dynamic registration."""

    _registry: ClassVar[dict[InputSourceType, Callable[[dict[str, Any]], InputSource]]] = {}

    @classmethod
    def register(
        cls, source_type: InputSourceType
    ) -> Callable[[Callable[[dict[str, Any]], InputSource]], Callable[[dict[str, Any]], InputSource]]:
        """Register an input source creator function.

        Parameters
        ----------
        source_type
            The input source type to register.

        Returns
        -------
        Decorator function for registration.

        Example
        -------
        >>> @InputSourceFactory.register(InputSourceType.LOCAL)
        >>> def create_local_source(config):
        >>>     return LocalInputSource(config)
        """

        def decorator(
            func: Callable[[dict[str, Any]], InputSource]
        ) -> Callable[[dict[str, Any]], InputSource]:
            cls._registry[source_type] = func
            return func

        return decorator

    @classmethod
    def create(cls, source_type: InputSourceType, config: dict[str, Any]) -> InputSource:
        """Create an input source instance.

        Parameters
        ----------
        source_type
            Type of input source to create.
        config
            Configuration dictionary for the input source.

        Returns
        -------
        InputSource instance.

        Raises
        ------
        ValueError
            If the input source type is not registered.
        """
        if source_type not in cls._registry:
            raise ValueError(
                f"Unknown input source type: {source_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        creator = cls._registry[source_type]
        return creator(config)

# Register implementations
InputSourceFactory.register(InputSourceType.LOCAL)(create_local_source)
InputSourceFactory.register(InputSourceType.S3)(create_s3_source)
