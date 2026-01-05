"""Qdrant filter builder implementation."""
from __future__ import annotations

import logging
from typing import Any, Optional

from .protocol import FilterBuilder

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


class QdrantFilterBuilder:
    """Filter builder for Qdrant vector store."""

    def __init__(self):
        """Initialize the Qdrant filter builder."""
        self._logger = logging.getLogger(__name__)

    def build(
        self,
        user_role: Optional[str] = None,
        user_tags: Optional[list[str]] = None,
        role_mapping: Optional[dict[str, list[str]]] = None,
    ) -> Optional[Any]:
        """Build Qdrant metadata filter for role-aware access control.

        Parameters
        ----------
        user_role : str, optional
            User's primary role.
        user_tags : list[str], optional
            User's direct access tags.
        role_mapping : dict[str, list[str]], optional
            Mapping of roles to authorized tags.

        Returns
        -------
        Filter or None
            Qdrant Filter object, or None if no filtering needed.
        """
        if not user_role and not user_tags:
            self._logger.debug("No role or tags provided - skipping access control filtering")
            return None

        authorized_tags: set[str] = set(user_tags or [])
        if user_role and role_mapping:
            role_tags = role_mapping.get(user_role, [])
            authorized_tags.update(role_tags)
            self._logger.debug(f"User role '{user_role}' expanded to tags: {role_tags}")

        should_conditions: list[FieldCondition] = []

        if user_role:
            should_conditions.append(
                FieldCondition(
                    key="metadata.required_role_strict", match=MatchValue(value=user_role)
                )
            )
            self._logger.debug(f"Added role filter: required_role_strict == '{user_role}'")

        if authorized_tags:
            should_conditions.append(
                FieldCondition(
                    key="metadata.access_tags", match=MatchAny(any=list(authorized_tags))
                )
            )
            self._logger.debug(f"Added tag filter: access_tags matches any of {authorized_tags}")

        if not should_conditions:
            self._logger.debug("No filter conditions generated")
            return None

        filter_obj = Filter(should=should_conditions)
        self._logger.info(
            f"Built Qdrant access filter for role='{user_role}', tags={authorized_tags}"
        )
        return filter_obj

