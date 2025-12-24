#!/usr/bin/env python3
"""Main script to ingest media files into the vector database."""

import logging
import sys
from pathlib import Path

from src import (
    ChunkerFactory,
    Config,
    EmbeddingModelFactory,
    Embeddings,
    LoaderFactory,
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
from src.logger import Logger
from src.pipeline import IngestionContext, PipelineExecutor, PipelineStatus
from src.pipeline.steps import (
    ChunkStep,
    EmbeddingGenerationStep,
    LoadStep,
    SaveStep,
)


def ingest_file(
    file_path: Path,
    config: Config,
    embedding_model: Embeddings,
    vector_store: VectorStore,
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
    """
    logger = logging.getLogger()
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {file_path}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    loader_name_str, loader_config_from_mapping = (
        LoaderHelper.get_loader_config_for_file(file_path, config)
    )
    file_extension = file_path.suffix.lower()

    try:
        loader_type = LoaderType(loader_name_str)
    except ValueError as exc:
        available = ", ".join(t.value for t in LoaderType)
        raise ValueError(
            f"Unknown loader type '{loader_name_str}' for file {file_path}. "
            f"Available loaders: {available}"
        ) from exc

    logger.info(
        f"Converting {file_extension} file to Markdown "
        f"(loader: {loader_name_str})..."
    )
    loader_config = LoaderHelper.create_loader_config(
        file_path,
        loader_name_str,
        loader_config_from_mapping,
        config,
    )
    loader = LoaderFactory.create(loader_type, **loader_config)

    # Build chunker config based on chunker type
    if config.chunking.chunker_name.value == "hybrid":
        # Hybrid chunker uses token-based parameters only
        chunker_config = {
            "max_tokens": config.chunking.max_tokens or 2000,
            "merge_peers": config.chunking.merge_peers,
        }
        # Pass google_api_key and model if available (for token counting)
        if config.embedding.embed_config and config.embedding.embed_config.get("google_api_key"):
            chunker_config["google_api_key"] = config.embedding.embed_config["google_api_key"]
        if config.embedding.embed_config and config.embedding.embed_config.get("model"):
            chunker_config["model"] = config.embedding.embed_config["model"]
    else:
        # Langchain chunker uses character-based parameters
        chunker_config = {
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "method": config.chunking.method,
        }
    chunker = ChunkerFactory.create(
        config.chunking.chunker_name,
        **chunker_config,
    )

    # Show chunking info based on chunker type
    if config.chunking.chunker_name.value == "hybrid":
        logger.info(
            f"Chunking markdown (hybrid, max_tokens: {config.chunking.max_tokens or 2000})..."
        )
    else:
        logger.info(
            f"Chunking markdown (size={config.chunking.chunk_size}, "
            f"overlap={config.chunking.chunk_overlap})..."
        )
    logger.info(f"Embedding chunks (model={config.embedding.embed_name})...")
    logger.info("Storing in vector database...")
    logger.info(f"  Database location: {config.vector_store.persist_directory}")
    logger.info(f"  Collection: {config.vector_store.collection_name}")

    context = IngestionContext(file_path=file_path)

    steps = [
        LoadStep(loader),
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
    logger.info(f"✓ Database location: {config.vector_store.persist_directory}")
    logger.info(f"✓ Collection: {config.vector_store.collection_name}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")


def main():
    """Run the ingestion pipeline for one or more media files."""
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger()

    input_path = config.paths.input_path

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
    logger.info(f"Database: {config.vector_store.persist_directory}")
    logger.info(f"Collection: {config.vector_store.collection_name}")
    # Show chunking info based on chunker type
    if config.chunking.chunker_name.value == "hybrid":
        logger.info(
            f"Chunker: hybrid (max_tokens: {config.chunking.max_tokens or 2000})"
        )
    else:
        logger.info(
            f"Chunk size: {config.chunking.chunk_size}, "
            f"overlap: {config.chunking.chunk_overlap}"
        )
    logger.info(f"Embedding model: {config.embedding.embed_name}")
    logger.info("")

    try:
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )

        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            persist_directory=str(config.vector_store.persist_directory),
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_model,
            **(config.vector_store.store_config or {}),
        )

        for media_file in media_files:
            ingest_file(media_file, config, embedding_model, vector_store)

        logger.info("✓ All files processed successfully.")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()

