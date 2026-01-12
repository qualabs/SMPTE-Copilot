"""Parallel ingestor for processing multiple files concurrently."""

import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from src.cli.constants import SEPARATOR_CHAR, SEPARATOR_LENGTH
from src.cli.models import IngestionResult


class ParallelIngestor:
    """Ingests multiple files in parallel using thread pools.

    This ingestor allows processing multiple files concurrently to improve
    throughput when ingesting large batches of documents. Uses threading
    which is ideal for I/O-bound workloads like file loading, API calls,
    and database operations.
    """

    def __init__(
        self,
        max_workers: int | None = None,
    ):
        """Initialize the parallel ingestor.

        Parameters
        ----------
        max_workers
            Maximum number of parallel workers. If None, uses the number of CPUs.
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    def execute(
        self,
        files: list[str],
        task_fn: Callable[[str, Any], IngestionResult],
        task_args: dict[str, Any],
    ) -> dict[str, list[IngestionResult]]:
        """Execute ingestion tasks in parallel.

        Parameters
        ----------
        files
            List of source identifiers (S3 URIs or local paths) to process.
        task_fn
            Function to execute for each file. Should accept (source_id, **task_args)
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

        # Determine actual number of workers that will be used
        cpu_count = os.cpu_count() or 1
        actual_workers = self.max_workers if self.max_workers is not None else cpu_count

        self.logger.info(f"Processing {len(files)} files in parallel using threading")
        self.logger.info(f"CPU cores available: {cpu_count}")
        self.logger.info(f"Workers to use: {actual_workers}")

        results = {"successful": [], "failed": []}
        futures: dict[Future, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for source_id in files:
                future = executor.submit(task_fn, source_id, **task_args)
                futures[future] = source_id

            # Process completed tasks as they finish
            total = len(files)

            for completed, future in enumerate(as_completed(futures), start=1):
                source_id = futures[future]

                try:
                    result = future.result()
                    if result.success:
                        results["successful"].append(result)
                        self.logger.info(
                            f"[{completed}/{total}] ✓ Successfully processed: {source_id}"
                        )
                    else:
                        results["failed"].append(result)
                        self.logger.error(
                            f"[{completed}/{total}] ✗ Failed to process: {source_id} - {result.error}"
                        )
                except Exception as e:
                    self.logger.info(f"[{completed}/{total}] ✗ Exception processing {source_id}, {e}")
                    results["failed"].append(
                        IngestionResult(
                            file_path=source_id,
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

