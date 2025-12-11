"""Context for query pipeline."""
from __future__ import annotations

from typing import Optional

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
