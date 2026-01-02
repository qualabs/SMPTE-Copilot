#!/usr/bin/env python3
"""Main script to ingest media files into the vector database."""

import argparse
import logging
import sys
from pathlib import Path

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
from src.cli.constants import (
    EXIT_CODE_ERROR,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.loaders.constants import SUPPORTED_FILE_EXTENSIONS
from src.loaders.helpers import LoaderHelper
from src.loaders.types import LoaderType
from src.chunkers.types import ChunkerType
from src.logger import Logger
from src.pipeline import IngestionContext, PipelineExecutor, PipelineStatus
from src.pipeline.steps import (
    ChunkStep,
    EmbeddingGenerationStep,
    LoadStep,
    PreprocessStep,
    SaveStep,
)


def ingest_file(
    file_path: Path,
    config: Config,
    embedding_model: Embeddings,
    vector_store: VectorStore,
    access_tags: list[str] = None,
    required_role: str = None,
) -> None:
    """Ingest a media file into the vector database using the pipeline pattern.

    Parameters
    ----------
    file_path
        Path to the media file to ingest.
    config
        Configuration object.
    embedding_model
        Embedding model instance.
    vector_store
        Vector store instance.
    access_tags
        Optional list of access control tags for the document.
    required_role
        Optional required role for strict access control.
    """
    logger = logging.getLogger(__name__)
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {file_path}")
    if access_tags:
        logger.info(f"Access tags: {access_tags}")
    if required_role:
        logger.info(f"Required role: {required_role}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

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

    chunker_config = config.chunking.chunker_config or {}
    embedding_config = config.embedding.embed_config or {}
    
    chunker = ChunkerFactory.create(
        config.chunking.chunker_name,
        **chunker_config,
        **embedding_config,
    )

    logger.info(f"Embedding chunks (model={config.embedding.embed_name})...")
    logger.info("Storing in vector database...")
    context = IngestionContext(file_path=file_path)
    
    # Set role-aware access control fields if provided
    if access_tags:
        context.access_tags = access_tags
    if required_role:
        context.required_role_strict = required_role

    preprocessing_config = config.preprocessing.preprocessing_config or {}
    preprocessor = PreprocessorFactory.create(
        config.preprocessing.preprocessing_name,
        **preprocessing_config,
    )

    steps = [
        LoadStep(loader),
        PreprocessStep(preprocessor),
        ChunkStep(chunker),
        EmbeddingGenerationStep(embedding_model, config.embedding.embed_name),
        SaveStep(vector_store),
    ]

    executor = PipelineExecutor(steps)
    context = executor.execute(context)

    if context.status == PipelineStatus.FAILED:
        raise RuntimeError(f"Pipeline failed: {context.error}")

    logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Ingestion Complete!")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"✓ File processed: {file_path.name}")
    if context.markdown_path:
        logger.info(f"✓ Markdown file: {context.markdown_path}")
    logger.info(f"✓ Chunks created: {len(context.chunks)}")
    logger.info(f"  Database location: {store_config_dict.get('persist_directory')}")
    logger.info(f"  Collection: {store_config_dict.get('collection_name')}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")


def main():
    """Run the ingestion pipeline for one or more media files."""
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger(__name__)

    input_path = config.paths.input_path
    
    # Get access control settings from config
    access_tags = config.access_control.default_access_tags or None
    required_role = config.access_control.default_required_role

    try:
        media_files = LoaderHelper.resolve_media_inputs(input_path)
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
        sys.exit(EXIT_CODE_ERROR)

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("RAG Media Ingestion Pipeline")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Inputs: {len(media_files)} file(s)")
    logger.info(f"Chunker: {config.chunking.chunker_name}")
    logger.info(f"Embedding model: {config.embedding.embed_name}")
    if access_tags:
        logger.info(f"Access tags: {access_tags}")
    if required_role:
        logger.info(f"Required role: {required_role}")
    logger.info("")

    try:
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )

        store_config = {
            "persist_directory": store_config.get("persist_directory"),
            "collection_name": store_config.get("collection_name"),
            "embedding_function": embedding_model,
        }    
        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            **store_config,
        )

        for media_file in media_files:
            ingest_file(
                media_file,
                config,
                embedding_model,
                vector_store,
                access_tags=access_tags if access_tags else None,
                required_role=required_role,
            )

        logger.info("✓ All files processed successfully.")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()

