"""Executes a sequence of pipeline steps sequentially."""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .status import PipelineStatus
from .step import PipelineStep

T = TypeVar("T", bound=BaseModel)


class PipelineExecutor:
    """Executes a sequence of pipeline steps sequentially.

    The executor stops execution if any step marks the context as failed.
    """

    def __init__(self, steps: list[PipelineStep[T]]):
        """Initialize the pipeline executor.

        Parameters
        ----------
        steps
            List of pipeline steps to execute in order.
        """
        self.steps = steps

    def execute(self, context: T) -> T:
        """Execute all pipeline steps sequentially.

        Parameters
        ----------
        context
            The initial context to pass through the pipeline.

        Returns
        -------
        The context after all steps have been executed (or stopped early).

        Raises
        ------
        Exception
            If any step raises an exception that is not handled.
        """
        context.mark_running()

        for step in self.steps:
            if context.status == PipelineStatus.FAILED:
                break

            try:
                step.run(context)
            except Exception as e:
                context.mark_failed(str(e))
                raise

        if context.status == PipelineStatus.RUNNING:
            context.mark_completed()

        return context
