from __future__ import annotations

"""Pipeline status enumeration."""

from enum import Enum


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
