"""Filter utilities for role-aware access control in retrieval."""
from __future__ import annotations

import logging
from typing import Any, Optional


def build_access_filter(
    user_role: Optional[str] = None,
    user_tags: Optional[list[str]] = None,
    role_mapping: Optional[dict[str, list[str]]] = None,
) -> Optional[Any]:
    """Build Qdrant metadata filter for role-aware access control.

    Implements the hybrid approach:
    (doc.required_role_strict == user_role) OR
    (doc.access_tags contains any of user_authorized_tags)

    Note: Qdrant supports native list matching - access_tags stored as Python list
    (e.g., ["Finance", "Public", "Internal"])

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
    Any or None
        Qdrant filter object, or None if no filtering needed.
    """
    logger = logging.getLogger(__name__)

    # No filtering if neither role nor tags provided
    if not user_role and not user_tags:
        logger.debug("No role or tags provided - skipping access control filtering")
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    except ImportError:
        logger.warning(
            "qdrant_client not installed - access control filtering requires Qdrant"
        )
        return None

    # Aggregate all authorized tags
    authorized_tags: set[str] = set(user_tags or [])
    if user_role and role_mapping:
        role_tags = role_mapping.get(user_role, [])
        authorized_tags.update(role_tags)
        logger.debug(
            f"User role '{user_role}' expanded to tags: {role_tags}"
        )

    # Build Qdrant filter with should (OR) conditions
    should_conditions: list[FieldCondition] = []

    # Add role match condition
    if user_role:
        should_conditions.append(
            FieldCondition(
                key="metadata.required_role_strict", match=MatchValue(value=user_role)
            )
        )
        logger.debug(f"Added role filter: required_role_strict == '{user_role}'")

    # Add tag match condition - native array matching!
    # In Qdrant with langchain-qdrant, metadata fields are under the metadata key
    if authorized_tags:
        should_conditions.append(
            FieldCondition(
                key="metadata.access_tags", match=MatchAny(any=list(authorized_tags))
            )
        )
        logger.debug(f"Added tag filter: access_tags matches any of {authorized_tags}")

    if not should_conditions:
        logger.debug("No filter conditions generated")
        return None

    # Return Qdrant Filter with should (OR) logic
    filter_obj = Filter(should=should_conditions)
    logger.info(
        f"Built access filter for role='{user_role}', tags={authorized_tags}"
    )
    return filter_obj
