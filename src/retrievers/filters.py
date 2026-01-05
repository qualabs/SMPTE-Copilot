"""Filter utilities for role-aware access control in retrieval."""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.retrievers.filters.factory import FilterBuilderFactory
from src.vector_stores.types import VectorStoreType


def build_access_filter(
    user_role: Optional[str] = None,
    user_tags: Optional[list[str]] = None,
    role_mapping: Optional[dict[str, list[str]]] = None,
    vector_store_type: Optional[VectorStoreType] = None,
) -> Optional[Any]:
    """Build metadata filter for role-aware access control.

    Implements the hybrid approach:
    (doc.required_role_strict == user_role) OR
    (doc.access_tags contains any of user_authorized_tags)

    The filter format is specific to the vector store type. If no vector store
    type is provided, returns None.

    Parameters
    ----------
    user_role : str, optional
        User's primary role.
    user_tags : list[str], optional
        User's direct access tags.
    role_mapping : dict[str, list[str]], optional
        Mapping of roles to authorized tags.
    vector_store_type : VectorStoreType, optional
        Type of vector store to build the filter for. If not provided, returns None.

    Returns
    -------
    Any or None
        Vector store-specific filter object, or None if no filtering needed
        or if vector store type is not provided.
    """
    logger = logging.getLogger(__name__)

    if not vector_store_type:
        logger.warning(
            "No vector store type provided - cannot build access filter. "
            "Access control filtering will be skipped."
        )
        return None

    if not user_role and not user_tags:
        logger.debug("No role or tags provided - skipping access control filtering")
        return None

    try:
        builder = FilterBuilderFactory.create(vector_store_type)
        return builder.build(
            user_role=user_role,
            user_tags=user_tags,
            role_mapping=role_mapping,
        )
    except ValueError as e:
        logger.warning(f"Could not create filter builder: {e}")
        return None
