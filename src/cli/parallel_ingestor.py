"""Parallel ingestor for processing multiple files concurrently."""

import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from src.cli.constants import SEPARATOR_CHAR, SEPARATOR_LENGTH
from src.cli.models import ExecutorType, IngestionResult


class ParallelIngestor:
    """Ingests multiple files in parallel using thread or process pools.

    This ingestor allows processing multiple files concurrently to improve
    throughput when ingesting large batches of documents.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        executor_type: ExecutorType = ExecutorType.THREAD,
    ):
        """Initialize the parallel ingestor.

        Parameters
        ----------
        max_workers
            Maximum number of parallel workers. If None, uses the number of CPUs.
        executor_type
            Type of executor to use (thread or process)
        """
        self.max_workers = max_workers
        self.executor_type = executor_type if isinstance(executor_type, ExecutorType) else ExecutorType(executor_type)
        self.logger = logging.getLogger(__name__)

    def execute(
        self,
        files: list[Path],
        task_fn: Callable[[Path, Any], IngestionResult],
        task_args: dict[str, Any],
    ) -> dict[str, list[IngestionResult]]:
        """Execute ingestion tasks in parallel.

        Parameters
        ----------
        files
            List of file paths to process.
        task_fn
            Function to execute for each file. Should accept (file_path, **task_args)
            and return an IngestionResult.
        task_args
            Additional arguments to pass to the task function.

        Returns
        -------
        dict with 'successful' and 'failed' lists of IngestionResult objects.
        """
        if not files:
            self.logger.warning("No files to process")
            return {"successful": [], "failed": []}

        executor_class = ThreadPoolExecutor if self.executor_type == ExecutorType.THREAD else ProcessPoolExecutor

        # Determine actual number of workers that will be used
        cpu_count = os.cpu_count() or 1
        actual_workers = self.max_workers if self.max_workers is not None else cpu_count

        self.logger.info(
            f"Processing {len(files)} files in parallel using {executor_class.__name__}"
        )
        self.logger.info(f"CPU cores available: {cpu_count}")
        self.logger.info(f"Workers to use: {actual_workers}")

        results = {"successful": [], "failed": []}
        futures: dict[Future, Path] = {}

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for file_path in files:
                future = executor.submit(task_fn, file_path, **task_args)
                futures[future] = file_path

            # Process completed tasks as they finish
            total = len(files)

            for completed, future in enumerate(as_completed(futures), start=1):
                file_path = futures[future]

                try:
                    result = future.result()
                    if result.success:
                        results["successful"].append(result)
                        self.logger.info(
                            f"[{completed}/{total}] ✓ Successfully processed: {file_path}"
                        )
                    else:
                        results["failed"].append(result)
                        self.logger.error(
                            f"[{completed}/{total}] ✗ Failed to process: {file_path} - {result.error}"
                        )
                except Exception as e:
                    self.logger.info(f"[{completed}/{total}] ✗ Exception processing {file_path}")
                    results["failed"].append(
                        IngestionResult(
                            file_path=file_path,
                            success=False,
                            error=str(e),
                        )
                    )

        return results

    def log_summary(self, results: dict[str, list[IngestionResult]]) -> None:
        """Log a summary of the ingestion results.

        Parameters
        ----------
        results
            Dictionary with 'successful' and 'failed' lists of results.
        """
        successful = results["successful"]
        failed = results["failed"]
        total = len(successful) + len(failed)

        self.logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        self.logger.info("Ingestion Summary")
        self.logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
        self.logger.info(f"Total files: {total}")
        self.logger.info(f"✓ Successful: {len(successful)}")
        self.logger.info(f"✗ Failed: {len(failed)}")

        if successful:
            total_chunks = sum(r.chunks_count for r in successful)
            self.logger.info(f"Total chunks created: {total_chunks}")

        if failed:
            self.logger.info("\nFailed files:")
            for result in failed:
                self.logger.info(f"  - {result.file_path}: {result.error}")

        self.logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")

