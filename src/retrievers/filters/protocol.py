"""Protocol for filter builders."""
from __future__ import annotations

from typing import Any, Optional, Protocol


class FilterBuilder(Protocol):
    """Protocol for building access control filters for vector stores.
    
    Each vector store implementation should provide its own filter builder
    that converts access control parameters into the appropriate filter format.
    """

    def build(
        self,
        user_role: Optional[str] = None,
        user_tags: Optional[list[str]] = None,
        role_mapping: Optional[dict[str, list[str]]] = None,
    ) -> Optional[Any]:
        """Build a metadata filter for role-aware access control.

        Parameters
        ----------
        user_role : str, optional
            User's primary role.
        user_tags : list[str], optional
            User's direct access tags.
        role_mapping : dict[str, list[str]], optional
            Mapping of roles to authorized tags.
        """
        ...

