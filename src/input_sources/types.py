from __future__ import annotations

"""Types for input source implementations."""

from enum import Enum


class InputSourceType(str, Enum):
    """Available input source types."""

    LOCAL = "local"
    S3 = "s3"
