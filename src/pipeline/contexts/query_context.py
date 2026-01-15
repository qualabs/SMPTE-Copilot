from __future__ import annotations

"""Context for query pipeline."""



from langchain_core.documents import Document
from pydantic import Field

from ..context import PipelineContext


class QueryContext(PipelineContext):
    """Context for query pipeline.

    Tracks the state of a query as it moves through the query pipeline:
    Embed Query -> Retrieve -> Generate
    """

    user_query: str
    query_vector: list[float] | None = None
    retrieved_docs: list[tuple[Document, float]] = Field(default_factory=list)
    prompt: str | None = None
    llm_response: str | None = None
    citations: list[dict] | None = None

    # Roles are automatically converted to tags via role_mapping
    user_role: str | None = None
    role_mapping: dict[str, list[str]] | None = None  # Role-to-tags mapping

    # Access control notification fields (used when notify_on_denied_access is enabled)
    restricted_docs: list[dict] = Field(default_factory=list)
    has_restricted_content: bool = False

