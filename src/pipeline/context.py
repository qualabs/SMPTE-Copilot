from __future__ import annotations

"""Base context for pipeline execution."""


from pydantic import BaseModel

from .status import PipelineStatus


class PipelineContext(BaseModel):
    """Base context for pipeline execution."""

    status: PipelineStatus = PipelineStatus.PENDING
    error: str | None = None

    def mark_failed(self, error: str) -> None:
        """Mark the context as failed with an error message."""
        self.status = PipelineStatus.FAILED
        self.error = error

    def mark_completed(self) -> None:
        """Mark the context as completed."""
        self.status = PipelineStatus.COMPLETED

    def mark_running(self) -> None:
        """Mark the context as running."""
        self.status = PipelineStatus.RUNNING
