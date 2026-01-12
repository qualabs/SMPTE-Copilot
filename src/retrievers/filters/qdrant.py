"""Qdrant filter builder implementation."""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny


class QdrantFilterBuilder:
    """Filter builder for Qdrant vector store."""

    def __init__(self):
        """Initialize the Qdrant filter builder."""
        self._logger = logging.getLogger(__name__)

    def build(
        self,
        user_role: str | None = None,
        role_mapping: dict[str, list[str]] | None = None,
    ) -> Any | None:
        """Build Qdrant metadata filter for tag-based access control.

        Roles are automatically converted to tags using role_mapping.
        Only filters by access_tags - simpler and more consistent model.

        Parameters
        ----------
        user_role : str, optional
            User's primary role (will be expanded to tags via role_mapping).
        role_mapping : dict[str, list[str]], optional
            Mapping of roles to authorized tags.

        Returns
        -------
        Filter or None
            Qdrant Filter object, or None if no filtering needed.
        """
        if not user_role or not role_mapping:
            self._logger.debug("No role or role_mapping provided - skipping access control filtering")
            return None

        authorized_tags = role_mapping.get(user_role, [])
        if not authorized_tags:
            self._logger.warning(
                f"User role '{user_role}' not found in role_mapping or has no authorized tags. "
                f"No access granted - returning filter that matches nothing."
            )
            return Filter(
                must=[
                    FieldCondition(
                        key="access_tags",
                        match=MatchAny(any=["__NO_ACCESS__"])
                    )
                ]
            )

        self._logger.debug(f"User role '{user_role}' expanded to tags: {authorized_tags}")

        filter_obj = Filter(
            must=[
                FieldCondition(
                    key="access_tags", match=MatchAny(any=authorized_tags)
                )
            ]
        )
        self._logger.info(
            f"Built Qdrant access filter for role '{user_role}' -> tags={authorized_tags}"
        )
        return filter_obj

