#!/usr/bin/env python3
"""Main script to ingest media files into the vector database."""

import logging
import sys
import time
from pathlib import Path

from src import (
    ChunkerFactory,
    Config,
    EmbeddingModelFactory,
    LoaderFactory,
    PreprocessorFactory,
    VectorStoreFactory,
)
from src.cli.constants import (
    EXIT_CODE_ERROR,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.cli.models import IngestionConfig, IngestionResult
from src.cli.parallel_ingestor import ParallelIngestor
from src.input_sources import InputSource, InputSourceFactory, InputSourceType
from src.loaders.constants import SUPPORTED_FILE_EXTENSIONS
from src.loaders.helpers import LoaderHelper
from src.logger import Logger
from src.pipeline import IngestionContext, PipelineExecutor, PipelineStatus
from src.pipeline.steps import (
    ChunkStep,
    EmbeddingGenerationStep,
    LoadStep,
    PreprocessStep,
    SaveStep,
)


def _build_ingestion_steps(
    file_path: Path,
    ingestion_config: IngestionConfig,
) -> list:
    """Build the list of pipeline steps based on configuration.

    Parameters
    ----------
    file_path
        Path to the media file to ingest.
    ingestion_config
        Ingestion configuration object.

    Returns
    -------
    list
        List of pipeline steps to execute.
    """
    logger = logging.getLogger(__name__)
    steps = []
    config = ingestion_config.config
    pipeline_config = config.pipeline.ingestion

    logger.info(f"Load step - enabled: {pipeline_config.load_enabled}")
    if pipeline_config.load_enabled:
        loader_name, loader_config_from_mapping = (
            LoaderHelper.get_loader_config_for_file(file_path, config)
        )
        file_extension = file_path.suffix.lower()
        logger.info(
            f"Converting {file_extension} file to Markdown "
            f"(loader: {loader_name})..."
        )
        loader_config = LoaderHelper.create_loader_config(
            file_path,
            loader_name,
            loader_config_from_mapping,
            config,
        )
        loader = LoaderFactory.create(loader_name, **loader_config)
        steps.append(LoadStep(loader))

    logger.info(f"Preprocess step - enabled: {pipeline_config.preprocess_enabled}")
    if pipeline_config.preprocess_enabled:
        preprocessing_config = config.preprocessing.preprocessing_config or {}
        preprocessor = PreprocessorFactory.create(
            config.preprocessing.preprocessing_name,
            **preprocessing_config,
        )
        steps.append(PreprocessStep(preprocessor))

    logger.info(f"Chunk step - enabled: {pipeline_config.chunk_enabled}")
    if pipeline_config.chunk_enabled:
        chunker_config = config.chunking.chunker_config or {}
        embedding_config = config.embedding.embed_config or {}
        chunker = ChunkerFactory.create(
            config.chunking.chunker_name,
            **chunker_config,
            **embedding_config,
        )
        steps.append(ChunkStep(chunker))

    logger.info(f"Save step - enabled: {pipeline_config.save_enabled}")
    if ingestion_config.embedding_model and ingestion_config.vector_store:
        logger.info(f"Embedding chunks (model={config.embedding.embed_name})...")
        steps.append(EmbeddingGenerationStep(ingestion_config.embedding_model, config.embedding.embed_name))
        logger.info("Storing in vector database...")
        steps.append(SaveStep(ingestion_config.vector_store))

    return steps


def ingest_file(
    source_id: str,
    ingestion_config: IngestionConfig,
) -> IngestionResult:
    """Ingest a media file into the vector database using the pipeline pattern.

    Parameters
    ----------
    source_id
        Source identifier (S3 URI or local file path).
    ingestion_config
        Ingestion configuration object.
    Returns
    -------
    IngestionResult
        Result object containing success status, chunks count, and any errors.
    """
    logger = logging.getLogger(__name__)
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {source_id}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    config = ingestion_config.config

    # Create InputSource instance for this worker
    input_source = _create_input_source_from_config(config)

    try:
        file_path_resolved = input_source.get_file(source_id)

        # Get access tags for this file based on folder mapping or default
        access_tags = config.access_control.get_tags_from_file(file_path_resolved)
        logger.info(f"Access tags for file: {access_tags}")

        context = IngestionContext(
            source_id=source_id,
            file_path=file_path_resolved,
            access_tags=access_tags,
        )

        # Build steps list dynamically based on pipeline configuration
        steps = _build_ingestion_steps(file_path_resolved, ingestion_config)

        executor = PipelineExecutor(steps)
        context = executor.execute(context)

        if context.status == PipelineStatus.FAILED:
            return IngestionResult(
                file_path=source_id,
                success=False,
                error=context.error,
            )

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Ingestion Complete!")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info(f"✓ File processed: {source_id}")
        if context.markdown_path:
            logger.info(f"✓ Markdown file: {context.markdown_path}")
        logger.info(f"✓ Chunks created: {len(context.chunks)}")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")

        return IngestionResult(
            file_path=source_id,
            success=True,
            chunks_count=len(context.chunks),
            markdown_path=context.markdown_path,
        )
    finally:
        # Clean up temporary files for this worker
        input_source.cleanup()


def _process_files_parallel(
    ingestion_config: IngestionConfig,
) -> bool:
    """Process files in parallel.

    Parameters
    ----------
    ingestion_config
        Ingestion configuration object.

    Returns
    -------
    bool
        True if all files processed successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Parallel processing enabled")

    pipeline_config = ingestion_config.config.pipeline.ingestion
    parallel_ingestor = ParallelIngestor(
        max_workers=pipeline_config.max_workers,
    )

    results = parallel_ingestor.execute(
        files=ingestion_config.source_ids,
        task_fn=ingest_file,
        task_args={
            "ingestion_config": ingestion_config,
        },
    )

    parallel_ingestor.log_summary(results)

    if results["failed"]:
        logger.error(f"✗ {len(results['failed'])} file(s) failed to process")
        return False

    logger.info("✓ All files processed successfully.")
    return True


def _process_files_sequential(
    ingestion_config: IngestionConfig,
) -> bool:
    """Process files sequentially.

    Parameters
    ----------
    ingestion_config
        Ingestion configuration object.

    Returns
    -------
    bool
        True if all files processed successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Sequential processing (parallel disabled)")

    failed_files = []

    for source_id in ingestion_config.source_ids:
        result = ingest_file(
            source_id,
            ingestion_config,
        )

        if not result.success:
            failed_files.append((source_id, result.error))

    if failed_files:
        logger.error("\nFailed files:")
        for source_id, error in failed_files:
            logger.error(f"  - {source_id}: {error}")
        logger.error(f"\n✗ {len(failed_files)} file(s) failed to process")
        return False

    logger.info("✓ All files processed successfully.")
    return True

def _create_input_source_from_config(config: Config) -> InputSource:
    """Create an InputSource instance from configuration.

    Parameters
    ----------
    config
        Configuration object containing input source settings.

    Returns
    -------
    InputSource
        Configured input source instance.
    """
    source_type = InputSourceType(config.input_source.get('source_type'))
    source_config = config.input_source.get('source_config') or {}
    input_path = config.paths.get('input_path')

    source_config = {**source_config, "base_path": str(input_path)}
    return InputSourceFactory.create(source_type, source_config)

def main():
    """Run the ingestion pipeline for one or more media files."""
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger(__name__)

    # Create input source and list files
    input_source = _create_input_source_from_config(config)
    source_ids = input_source.list_files("", list(SUPPORTED_FILE_EXTENSIONS))
    input_source.cleanup()

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("RAG Media Ingestion Pipeline")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Using input source: {config.input_source.get('source_type')}")
    logger.info(f"Source config: {config.input_source.get('source_config')}")
    logger.info(f"Inputs: {len(source_ids)} file(s)")
    logger.info(f"Chunker: {config.chunking.chunker_name}")
    logger.info(f"Embedding model: {config.embedding.embed_name}")
    logger.info(f"Database location: {config.vector_store.store_config.get('persist_directory')}")
    logger.info(f"Collection: {config.vector_store.store_config.get('collection_name')}")
    logger.info(f"Access tags: {config.access_control.get('default_access_tags')}")

    try:
        pipeline_config = config.pipeline.ingestion
        embedding_model = None
        vector_store = None

        # Only create embedding_model and vector_store if save step is enabled
        if pipeline_config.save_enabled:
            embedding_model = EmbeddingModelFactory.create(
                config.embedding.embed_name,
                **(config.embedding.embed_config or {}),
            )

            store_config = {
                "embedding_function": embedding_model,
                **(config.vector_store.store_config or {}),
            }
            vector_store = VectorStoreFactory.create(
                config.vector_store.store_name,
                **store_config,
            )

        # Create ingestion configuration
        ingestion_config = IngestionConfig(
            config=config,
            source_ids=source_ids,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Process files in parallel if enabled, otherwise sequentially
        start_time = time.time()
        process_fn = _process_files_parallel if pipeline_config.parallel_enabled else _process_files_sequential
        process_fn(ingestion_config)
        elapsed_time = time.time() - start_time

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Ingestion complete!")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"{len(source_ids)} file(s) processed successfully")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()

