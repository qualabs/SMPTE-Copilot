#!/usr/bin/env python3
"""Main script to ingest media files into the vector database."""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from src import (
    LoaderFactory,
    ChunkerFactory,
    EmbeddingModelFactory,
    VectorStore,
    VectorStoreFactory,
    Config,
    Embeddings,
)
from src.loaders.types import LoaderType
from src.loaders.helpers import LoaderHelper
from src.vector_stores.helpers import VectorStoreHelper
from src.embeddings.helpers import EmbeddingHelper
from src.cli.constants import (
    SEPARATOR_LENGTH,
    SEPARATOR_CHAR,
    EXIT_CODE_ERROR,
)
from src.logger import Logger
import logging

def ingest_file(
    file_path: Path,
    vector_store: VectorStore,
    embedding_model: Embeddings,
    config: Config,
) -> None:
    """Ingest a media file into the vector database.
    
    Parameters
    ----------
    file_path
        Path to the media file to ingest.
    vector_store
        Vector store instance.
    embedding_model
        Embedding model instance.
    config
        Configuration object.
    """
    logger = logging.getLogger()
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {file_path}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    # Determine loader based on file extension and configuration
    loader_name_str, loader_config_from_mapping = LoaderHelper.get_loader_config_for_file(file_path, config)
    file_extension = file_path.suffix.lower()
    
    # Convert string to LoaderType enum
    try:
        loader_type = LoaderType(loader_name_str)
    except ValueError:
        available = ", ".join(t.value for t in LoaderType)
        raise ValueError(
            f"Unknown loader type '{loader_name_str}' for file {file_path}. "
            f"Available loaders: {available}"
        )
    
    # Step 1: Media → Text/Markdown
    logger.info(f"Step 1: Converting {file_extension} file to Markdown (loader: {loader_name_str})...")
    loader_config = LoaderHelper.create_loader_config(
        file_path,
        loader_name_str,
        loader_config_from_mapping,
        config,
    )
    loader = LoaderFactory.create(loader_type, **loader_config)
    markdown_path = loader.to_markdown_file()
    logger.info(f"✓ Markdown saved to: {markdown_path}")

    # Step 2: Markdown → Chunks
    logger.info(
        f"\nStep 2: Chunking markdown (size={config.chunking.chunk_size}, "
        f"overlap={config.chunking.chunk_overlap})..."
    )
    chunker_config = {
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "method": config.chunking.method,
    }
    chunker = ChunkerFactory.create(
        config.chunking.chunker_name,
        **chunker_config,
    )
    chunks = chunker.chunk_markdown_file(str(markdown_path))
    logger.info(f"✓ Created {len(chunks)} chunks")

    # Step 3: Chunks → Embeddings
    logger.info(f"\nStep 3: Embedding chunks (model={config.embedding.embed_name})...")
    embedded_chunks = EmbeddingHelper.embed_chunks(embedding_model, chunks, model_name=config.embedding.embed_name)
    logger.info(f"✓ Embedded {len(embedded_chunks)} chunks")

    # Step 4: Embeddings → Vector Database
    logger.info(f"\nStep 4: Storing in vector database...")
    logger.info(f"  Database location: {config.vector_store.persist_directory}")
    logger.info(f"  Collection: {config.vector_store.collection_name}")

    VectorStoreHelper.ingest_chunks_with_embeddings(vector_store, embedded_chunks)
    vector_store.persist()
    logger.info(f"✓ Ingested {len(embedded_chunks)} chunks")

    # Summary
    logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Ingestion Complete!")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"✓ File processed: {file_path.name}")
    logger.info(f"✓ Markdown file: {markdown_path}")
    logger.info(f"✓ Chunks created: {len(chunks)}")
    logger.info(f"✓ Database location: {config.vector_store.persist_directory}")
    logger.info(f"✓ Collection: {config.vector_store.collection_name}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")


def main():
    """Run the ingestion pipeline for one or more media files."""
    # Load configuration
    config = Config.get_config()
    
    # Setup logging from config
    Logger.setup(config)
    logger = logging.getLogger()
    
    # Determine input (CLI arg > config > default path)
    input_path = config.paths.input_path

    try:
        media_files = LoaderHelper.resolve_media_inputs(input_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"✗ {exc}")
        logger.error("\nUsage:")
        logger.error("  python ingest.py ./data  # Ingest all supported files in directory")
        from src.loaders.constants import SUPPORTED_FILE_EXTENSIONS
        supported_types = ", ".join(SUPPORTED_FILE_EXTENSIONS)
        logger.error(f"\nSupported file types: {supported_types}")
        logger.error("\nConfiguration:")
        logger.error("  - Config file: config.yaml or config.yml")
        logger.error("  - Or set RAG_CONFIG_FILE=/path/to/config.yaml")
        logger.error("  - Default paths are relative to current working directory")
        sys.exit(EXIT_CODE_ERROR)

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("RAG Media Ingestion Pipeline")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Inputs: {len(media_files)} file(s)")
    logger.info(f"Database: {config.vector_store.persist_directory}")
    logger.info(f"Collection: {config.vector_store.collection_name}")
    logger.info(f"Chunk size: {config.chunking.chunk_size}, overlap: {config.chunking.chunk_overlap}")
    logger.info(f"Embedding model: {config.embedding.embed_name}")
    logger.info("")

    try:
        # Create embedding model using factory
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )
        
        # Create vector store using factory
        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            persist_directory=str(config.vector_store.persist_directory),
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_model,
            **(config.vector_store.store_config or {}),
        )

        for media_file in media_files:
            ingest_file(media_file, vector_store, embedding_model, config)

        logger.info("✓ All files processed successfully.")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()

