from __future__ import annotations

"""Context for query pipeline."""


from typing import Optional

from langchain.schema import Document
from pydantic import Field

from ..context import PipelineContext


class QueryContext(PipelineContext):
    """Context for query pipeline.

    Tracks the state of a query as it moves through the query pipeline:
    Embed Query -> Retrieve -> Generate
    """

    user_query: str
    query_vector: Optional[list[float]] = None
    retrieved_docs: list[tuple[Document, float]] = Field(default_factory=list)
    prompt: Optional[str] = None
    llm_response: Optional[str] = None
    citations: Optional[list[dict]] = None

    # Roles are automatically converted to tags via role_mapping
    user_role: Optional[str] = None
    role_mapping: Optional[dict[str, list[str]]] = None  # Role-to-tags mapping

