#!/usr/bin/env python3
"""Main script to ingest media files into the vector database."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src import (
    ChunkerFactory,
    Config,
    EmbeddingModelFactory,
    Embeddings,
    LoaderFactory,
    PreprocessorFactory,
    VectorStore,
    VectorStoreFactory,
)
from src.input_sources import InputSourceFactory, InputSourceType
from src.cli.constants import (
    EXIT_CODE_ERROR,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.cli.models import IngestionResult
from src.cli.parallel_ingestor import ParallelIngestor
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
    config: Config,
    embedding_model: Optional[Embeddings],
    vector_store: Optional[VectorStore],
) -> list:
    """Build the list of pipeline steps based on configuration.

    Parameters
    ----------
    file_path
        Path to the media file to ingest.
    config
        Configuration object.
    embedding_model
        Embedding model instance (can be None if save step is disabled).
    vector_store
        Vector store instance (can be None if save step is disabled).

    Returns
    -------
    list
        List of pipeline steps to execute.
    """
    logger = logging.getLogger(__name__)
    steps = []
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
    if embedding_model and vector_store:
        logger.info(f"Embedding chunks (model={config.embedding.embed_name})...")
        steps.append(EmbeddingGenerationStep(embedding_model, config.embedding.embed_name))
        logger.info("Storing in vector database...")
        steps.append(SaveStep(vector_store))

    return steps


def ingest_file(
    file_path: Path,
    config: Config,
    embedding_model: Optional[Embeddings],
    vector_store: Optional[VectorStore],
    access_tags: Optional[list[str]] = None,
) -> IngestionResult:
    """Ingest a media file into the vector database using the pipeline pattern.

    Parameters
    ----------
    file_path
        Path to the media file to ingest.
    config
        Configuration object.
    embedding_model
        Embedding model instance (can be None if save step is disabled).
    vector_store
        Vector store instance (can be None if save step is disabled).
    access_tags
        Optional list of access control tags for the document.

    Returns
    -------
    IngestionResult
        Result object containing success status, chunks count, and any errors.
    """
    logger = logging.getLogger(__name__)
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {file_path}")
    if access_tags:
        logger.info(f"Access tags: {access_tags}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    context = IngestionContext(file_path=file_path)

    # Set tag-based access control if provided
    if access_tags:
        context.access_tags = access_tags

    # Build steps list dynamically based on pipeline configuration
    steps = _build_ingestion_steps(file_path, config, embedding_model, vector_store)

    executor = PipelineExecutor(steps)
    context = executor.execute(context)

    if context.status == PipelineStatus.FAILED:
        return IngestionResult(
            file_path=file_path,
            success=False,
            error=context.error,
        )

    logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Ingestion Complete!")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"✓ File processed: {file_path.name}")
    if context.markdown_path:
        logger.info(f"✓ Markdown file: {context.markdown_path}")
    logger.info(f"✓ Chunks created: {len(context.chunks)}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")

    return IngestionResult(
        file_path=file_path,
        success=True,
        chunks_count=len(context.chunks),
        markdown_path=context.markdown_path,
    )


def _process_files_parallel(
    media_files: list[Path],
    config: Config,
    embedding_model: Optional[Embeddings],
    vector_store: Optional[VectorStore],
    access_tags: Optional[list[str]],
) -> bool:
    """Process files in parallel.

    Parameters
    ----------
    media_files
        List of media files to process.
    config
        Configuration object.
    embedding_model
        Embedding model instance.
    vector_store
        Vector store instance.
    access_tags
        Optional access control tags.

    Returns
    -------
    bool
        True if all files processed successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Parallel processing enabled")

    pipeline_config = config.pipeline.ingestion
    parallel_ingestor = ParallelIngestor(
        max_workers=pipeline_config.max_workers,
        executor_type=pipeline_config.executor_type,
    )

    results = parallel_ingestor.execute(
        files=media_files,
        task_fn=ingest_file,
        task_args={
            "config": config,
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "access_tags": access_tags,
        },
    )

    parallel_ingestor.log_summary(results)

    if results["failed"]:
        logger.error(f"✗ {len(results['failed'])} file(s) failed to process")
        return False

    logger.info("✓ All files processed successfully.")
    return True


def _process_files_sequential(
    media_files: list[Path],
    config: Config,
    embedding_model: Optional[Embeddings],
    vector_store: Optional[VectorStore],
    access_tags: Optional[list[str]],
) -> bool:
    """Process files sequentially.

    Parameters
    ----------
    media_files
        List of media files to process.
    config
        Configuration object.
    embedding_model
        Embedding model instance.
    vector_store
        Vector store instance.
    access_tags
        Optional access control tags.

    Returns
    -------
    bool
        True if all files processed successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Sequential processing (parallel disabled)")

    failed_files = []

    for media_file in media_files:
        result = ingest_file(
            media_file,
            config,
            embedding_model,
            vector_store,
            access_tags=access_tags if access_tags else None,
        )

        if not result.success:
            failed_files.append((media_file, result.error))

    if failed_files:
        logger.error("\nFailed files:")
        for file_path, error in failed_files:
            logger.error(f"  - {file_path.name}: {error}")
        logger.error(f"\n✗ {len(failed_files)} file(s) failed to process")
        return False

    logger.info("✓ All files processed successfully.")
    return True


def main():
    """Run the ingestion pipeline for one or more media files."""
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger(__name__)

    input_path = config.paths.input_path

    # Get access control settings from config
    access_tags = config.access_control.default_access_tags or None

    # Initialize input source
    input_source = None
    try:
        source_type = InputSourceType(config.input_source.source_type)
        source_config = config.input_source.source_config or {}
        input_source = InputSourceFactory.create(source_type, source_config)
        logger.info(f"Using input source: {source_type.value}")
    except ValueError as e:
        logger.error(f"✗ Invalid input source type: {config.input_source.source_type}")
        logger.error(f"Available types: {[t.value for t in InputSourceType]}")
        sys.exit(EXIT_CODE_ERROR)
    except Exception as e:
        logger.exception(f"✗ Error initializing input source: {e}")
        sys.exit(EXIT_CODE_ERROR)

    try:
        # List files using input source
        file_ids = input_source.list_files(str(input_path), list(SUPPORTED_FILE_EXTENSIONS))
        
        if not file_ids:
            logger.warning(f"No supported files found in: {input_path}")
            supported_types = ", ".join(SUPPORTED_FILE_EXTENSIONS)
            logger.warning(f"Supported file types: {supported_types}")
            sys.exit(EXIT_CODE_ERROR)
        
        # Get actual file paths (for local: same as file_ids, for S3: downloads to temp)
        media_files = [input_source.get_file(file_id) for file_id in file_ids]
        
    except (FileNotFoundError, ValueError):
        logger.exception("✗ Error resolving media inputs")
        logger.exception("\nUsage:")
        logger.exception("  python ingest.py ./data  # Ingest all supported files in directory")
        supported_types = ", ".join(SUPPORTED_FILE_EXTENSIONS)
        logger.exception(f"\nSupported file types: {supported_types}")
        logger.exception("\nConfiguration:")
        logger.exception("  - Config file: config.yaml or config.yml")
        logger.exception("  - Or set RAG_CONFIG_FILE=/path/to/config.yaml")
        logger.exception("  - Default paths are relative to current working directory")
        if input_source:
            input_source.cleanup()
        sys.exit(EXIT_CODE_ERROR)

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("RAG Media Ingestion Pipeline")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Inputs: {len(media_files)} file(s)")
    logger.info(f"Chunker: {config.chunking.chunker_name}")
    logger.info(f"Embedding model: {config.embedding.embed_name}")
    logger.info(f"Database location: {config.vector_store.store_config.get('persist_directory')}")
    logger.info(f"Collection: {config.vector_store.store_config.get('collection_name')}")

    if access_tags:
        logger.info(f"Access tags: {access_tags}")
    logger.info("")

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

        # Process files in parallel if enabled, otherwise sequentially
        start_time = time.time()
        process_fn = _process_files_parallel if pipeline_config.parallel_enabled else _process_files_sequential
        process_fn(
            media_files,
            config,
            embedding_model,
            vector_store,
            access_tags,
        )
        elapsed_time = time.time() - start_time

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Ingestion complete!")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"{len(media_files)} file(s) processed successfully")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)
    finally:
        # Clean up temporary files (S3 downloads, etc.)
        if input_source:
            input_source.cleanup()


if __name__ == "__main__":
    main()

