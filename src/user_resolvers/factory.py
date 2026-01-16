"""Factory for creating user role resolvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .json_resolver import create_json_resolver
from .protocol import UserRoleResolver
from .types import UserResolverType


class UserRoleResolverFactory:
    """Factory for creating user role resolvers.

    Supports registration of new resolver types for extensibility.
    """

    _registry: ClassVar[dict[UserResolverType, Callable[[dict[str, Any]], UserRoleResolver]]] = {}

    @classmethod
    def register(cls, resolver_type: UserResolverType):
        """Register a new user role resolver factory.

        Parameters
        ----------
        resolver_type
            Type to register the resolver under.

        Returns
        -------
        Callable
            Decorator that registers the factory function.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], UserRoleResolver]):
            cls._registry[resolver_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, resolver_type: UserResolverType, **kwargs: Any) -> UserRoleResolver:
        """Create a user role resolver by type.

        Parameters
        ----------
        resolver_type
            Type of the resolver to create.
        **kwargs
            Additional arguments passed to the resolver factory.

        Returns
        -------
        UserRoleResolver
            Configured resolver instance.

        Raises
        ------
        ValueError
            If the resolver type is not registered.
        """
        if resolver_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown resolver type: {resolver_type}. "
                f"Available types: {available}"
            )
        return cls._registry[resolver_type](kwargs)


UserRoleResolverFactory.register(UserResolverType.JSON)(create_json_resolver)
