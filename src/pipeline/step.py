"""Protocol for pipeline step implementations."""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PipelineStep(Protocol[T]):
    """Protocol for pipeline step implementations.

    Any class implementing the run method can be used as a pipeline step.
    This allows swapping step implementations without changing the rest of the code.
    """

    def run(self, context: T) -> None:
        """Execute the pipeline step.

        Parameters
        ----------
        context
            The pipeline context to operate on and modify.

        Raises
        ------
        Exception
            If the step fails and cannot recover, it should raise an exception
            or mark the context as failed.
        """
        ...
