"""User role resolver module for role-based access control."""

from .factory import UserRoleResolverFactory
from .json_resolver import JsonUserRoleResolver
from .protocol import UserRoleResolver
from .types import UserResolverType

__all__ = [
    "JsonUserRoleResolver",
    "UserResolverType",
    "UserRoleResolver",
    "UserRoleResolverFactory",
]
