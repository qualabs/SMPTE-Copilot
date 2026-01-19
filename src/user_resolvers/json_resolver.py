"""JSON-based user role resolver implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils import load_json_file

logger = logging.getLogger(__name__)


class JsonUserRoleResolver:
    """Resolves user roles from a JSON mapping file.

    The JSON file should have the following structure:
    {
        "users": {
            "user@example.com": "Finance",
            "admin@example.com": "Admin"
        },
        "default_role": "Public"
    }

    Users are looked up by email address. If the user is not found,
    the default_role from the file is used (or the constructor default).
    """

    def __init__(
        self,
        mapping_file: Path | str | None = None,
        default_role: str = "Public",
    ) -> None:
        """Initialize the JSON user role resolver.

        Parameters
        ----------
        mapping_file
            Path to the JSON file containing user-to-role mappings.
            If None or file doesn't exist, all users get the default role.
        default_role
            Default role to use when user is not found in the mapping.
            Can be overridden by 'default_role' in the JSON file.
        """
        self._default_role = default_role
        self._users: dict[str, str] = {}

        if mapping_file:
            self._load_mapping(Path(mapping_file))

    def _load_mapping(self, mapping_file: Path) -> None:
        """Load the user mapping from a JSON file.

        Parameters
        ----------
        mapping_file
            Path to the JSON mapping file.
        """
        logger.info(f"Loading user mapping from: {mapping_file}")
        data = load_json_file(mapping_file)

        if data is None:
            return

        self._users = data.get("users", {})

        logger.info(
            f"Loaded {len(self._users)} user mappings, "
            f"default_role='{self._default_role}'"
        )

    def resolve_role(
        self,
        user_email: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Resolve the role for a user based on email.

        Parameters
        ----------
        user_email
            User's email address to look up.
        user_id
            User's ID (not used in this implementation, but part of protocol).

        Returns
        -------
        str
            The user's role if found, otherwise the default role.
        """
        if user_email and user_email in self._users:
            role = self._users[user_email]
            logger.debug(f"Resolved role for '{user_email}': {role}")
            return role

        logger.debug(
            f"User '{user_email or user_id}' not found, using default role: {self._default_role}"
        )
        return self._default_role

    @property
    def default_role(self) -> str:
        """Get the default role used when user is not found.

        Returns
        -------
        str
            The default role name.
        """
        return self._default_role


def create_json_resolver(config: dict[str, Any]) -> JsonUserRoleResolver:
    """Factory function to create a JsonUserRoleResolver.

    Parameters
    ----------
    config
        Configuration dictionary with optional keys:
        - mapping_file: Path to the JSON mapping file
        - default_role: Default role when user not found

    Returns
    -------
    JsonUserRoleResolver
        Configured resolver instance.
    """
    return JsonUserRoleResolver(
        mapping_file=config.get("mapping_file"),
        default_role=config.get("default_role", "Public"),
    )
