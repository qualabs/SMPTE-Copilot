"""Protocol for filter builders."""
from __future__ import annotations

from typing import Any, Optional, Protocol


class FilterBuilder(Protocol):
    """Protocol for building access control filters for vector stores.

    Each vector store implementation should provide its own filter builder
    that converts access control parameters into the appropriate filter format.

    The system uses tag-based filtering exclusively. Roles are automatically
    converted to tags using role_mapping for a simpler, more consistent model.
    """

    def build(
        self,
        user_role: Optional[str] = None,
        role_mapping: Optional[dict[str, list[str]]] = None,
    ) -> Optional[Any]:
        """Build a metadata filter for tag-based access control.

        Roles are automatically expanded to tags using role_mapping.
        Only filters by access_tags for consistency.

        Parameters
        ----------
        user_role : str, optional
            User's primary role (will be expanded to tags via role_mapping).
        role_mapping : dict[str, list[str]], optional
            Mapping of roles to authorized tags.
        """
        ...

