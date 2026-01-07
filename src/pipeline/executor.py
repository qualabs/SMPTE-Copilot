from __future__ import annotations

"""Executes a sequence of pipeline steps sequentially."""

import logging
import time
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
        logger = logging.getLogger(__name__)
        context.mark_running()

        for step in self.steps:
            if context.status == PipelineStatus.FAILED:
                break

            step_name = step.__class__.__name__
            start_time = time.time()

            try:
                logger.info(f"Starting step: {step_name}")
                step.run(context)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed step: {step_name} (took {elapsed_time:.2f}s)")
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.exception(f"Failed step: {step_name} after {elapsed_time:.2f}s")
                context.mark_failed(str(e))
                raise

        if context.status == PipelineStatus.RUNNING:
            context.mark_completed()

        return context
