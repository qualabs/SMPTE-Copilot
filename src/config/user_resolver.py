"""User resolver configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings

from src.user_resolvers.types import UserResolverType


class UserResolverConfig(BaseSettings):
    """Configuration for user role resolution.

    This configuration controls how user roles are determined
    from identity information (email, user ID, etc.) received
    from external systems like OpenWebUI.

    Note: The default_role is taken from access_control.default_user_role
    to avoid duplication.
    """

    resolver_name: UserResolverType = Field(
        default=UserResolverType.JSON,
        description="User resolver type (json, database, ldap)",
    )

    resolver_config: dict | None = Field(
        default=None,
        description="Resolver-specific configuration (e.g., mapping_file for JSON resolver)",
    )
