"""Step that applies access control and tracks restricted documents."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from ..contexts.query_context import QueryContext


class AccessControlStep:
    """Step that separates accessible and restricted documents based on user role.

    This step is used when notify_on_denied_access is enabled. Instead of filtering
    at the database level, it retrieves all matching documents and separates them
    into accessible (user can view) and restricted (user cannot view but is notified).
    """

    def __init__(self) -> None:
        """Initialize the access control step."""
        self._logger = logging.getLogger(__name__)

    def run(self, context: QueryContext) -> None:
        """Separate retrieved documents into accessible and restricted.

        Parameters
        ----------
        context
            Query context with retrieved_docs populated.
        """
        if not self._has_access_control(context):
            self._logger.debug("No access control configured, skipping separation")
            return

        authorized_tags = self._get_authorized_tags(context)
        if not authorized_tags:
            self._handle_no_authorized_tags(context)
            return

        accessible, restricted = self._separate_documents(
            context.retrieved_docs, authorized_tags
        )

        context.retrieved_docs = accessible
        context.restricted_docs = restricted
        context.has_restricted_content = len(restricted) > 0

    def _has_access_control(self, context: QueryContext) -> bool:
        """Check if access control is configured."""
        return bool(context.user_role and context.role_mapping)

    def _get_authorized_tags(self, context: QueryContext) -> list[str]:
        """Get the list of tags the user is authorized to access."""
        return context.role_mapping.get(context.user_role, [])

    def _handle_no_authorized_tags(self, context: QueryContext) -> None:
        """Handle case where user role has no authorized tags."""
        self._logger.warning(
            f"User role '{context.user_role}' has no authorized tags. "
            "All documents will be marked as restricted."
        )
        restricted = []
        for doc, _ in context.retrieved_docs:
            metadata = doc.metadata or {}
            restricted.append({
                "source": metadata.get("source") or metadata.get("file_path"),
                "page": metadata.get("page") or metadata.get("page_number"),
                "required_tags": metadata.get("access_tags", []),
            })
        context.retrieved_docs = []
        context.restricted_docs = restricted
        context.has_restricted_content = len(restricted) > 0

    def _separate_documents(
        self,
        docs_with_scores: list[tuple[Document, float]],
        authorized_tags: list[str],
    ) -> tuple[list[tuple[Document, float]], list[dict]]:
        """Separate documents into accessible and restricted based on tags.

        Returns
        -------
        tuple
            (accessible_docs_with_scores, restricted_doc_metadata)
        """
        accessible: list[tuple[Document, float]] = []
        restricted: list[dict] = []

        for doc, score in docs_with_scores:
            if self._is_document_accessible(doc, authorized_tags):
                accessible.append((doc, score))
            else:
                metadata = doc.metadata or {}
                doc_tags = metadata.get("access_tags", [])
                restricted.append({
                    "source": metadata.get("source") or metadata.get("file_path"),
                    "page": metadata.get("page") or metadata.get("page_number"),
                    "required_tags": doc_tags,
                })

        self._logger.info(
            f"Access control: {len(accessible)} accessible, {len(restricted)} restricted"
        )
        return accessible, restricted

    def _is_document_accessible(
        self, doc: Document, authorized_tags: list[str]
    ) -> bool:
        """Check if a document is accessible based on its tags."""
        doc_tags = doc.metadata.get("access_tags", [])
        return any(tag in authorized_tags for tag in doc_tags)
