"""Protocol for user role resolver implementations."""

from __future__ import annotations

from typing import Protocol


class UserRoleResolver(Protocol):
    """Protocol for resolving user roles from identity information.

    Implementations should resolve a user's role based on their email,
    user ID, or other identity information. This enables role-based
    access control where different users see different document sets.
    """

    def resolve_role(
        self,
        user_email: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Resolve the role for a user based on email or ID.

        Parameters
        ----------
        user_email
            User's email address (e.g., from OpenWebUI header).
        user_id
            User's unique identifier (e.g., from OpenWebUI header).

        Returns
        -------
        str
            The resolved role name. Returns default_role if user not found.
        """
        ...

    @property
    def default_role(self) -> str:
        """Get the default role used when user is not found.

        Returns
        -------
        str
            The default role name.
        """
        ...
