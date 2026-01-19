"""User resolver types."""

from enum import Enum


class UserResolverType(str, Enum):
    """User resolver type enumeration."""

    JSON = "json"
