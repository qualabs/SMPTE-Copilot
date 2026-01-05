from __future__ import annotations

"""Context for query pipeline."""

from typing import List, Optional

from langchain.schema import Document

from ..context import PipelineContext


class QueryContext(PipelineContext):
    """Context for query pipeline.

    Tracks the state of a query as it moves through the query pipeline:
    Embed Query -> Retrieve -> Generate
    """

    user_query: str
    query_vector: Optional[list[float]] = None
    retrieved_docs: list[tuple[Document, float]] = []
    prompt: Optional[str] = None
    llm_response: Optional[str] = None
    citations: Optional[List[dict]] = None
    
    # Role-aware access control fields (optional)
    user_role: Optional[str] = None  # User role for access control
    user_tags: list[str] = []  # User tags for access control
    role_mapping: Optional[dict[str, list[str]]] = None  # Role-to-tags mapping

