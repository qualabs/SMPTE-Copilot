#!/usr/bin/env python3
"""Main script to ingest PDF documents into the vector database."""

import sys
from pathlib import Path
from typing import List
from src import (
    LoaderFactory,
    ChunkerFactory,
    EmbeddingModelFactory,
    VectorStore,
    VectorStoreFactory,
    Config,
)
from src.vector_stores.helpers import VectorStoreHelper
from src.embeddings.helpers import EmbeddingHelper
from src.cli.constants import (
    SEPARATOR_LENGTH,
    SEPARATOR_CHAR,
    EXIT_CODE_ERROR,
)
from src.logger import Logger
import logging

def _resolve_pdf_inputs(input_path: Path) -> List[Path]:
    """Resolve input path to a list of PDF files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {input_path}")
        return [input_path]
    # Directory: collect all PDFs (non-recursive)
    pdf_files = sorted(input_path.glob(f"*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {input_path}")
    return pdf_files


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ingest_pdf(
    pdf_path: Path,
    vector_store: VectorStore,
    embedding_model,
    config,
) -> None:
    logger = logging.getLogger()
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Ingesting: {pdf_path}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    # Step 1: PDF → Markdown
    logger.info("Step 1: Converting PDF to Markdown...")
    loader_config = {
        "pdf_path": str(pdf_path),
        "output_dir": str(_prepare_output_dir(config.paths.markdown_dir)),
    }
    loader = LoaderFactory.create(
        config.loader.loader_name,
        **loader_config,
    )
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
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    logger.info(f"✓ Ingested {len(embedded_chunks)} chunks")

    # Summary
    logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Ingestion Complete!")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"✓ PDF processed: {pdf_path.name}")
    logger.info(f"✓ Markdown file: {markdown_path}")
    logger.info(f"✓ Chunks created: {len(chunks)}")
    logger.info(f"✓ Database location: {config.vector_store.persist_directory}")
    logger.info(f"✓ Collection: {config.vector_store.collection_name}")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH + "\n")


def main():
    """Run the ingestion pipeline for one or more PDFs."""
    # Load configuration
    config = Config.get_config()
    
    # Setup logging from config
    Logger.setup(config)
    logger = logging.getLogger()
    
    # Determine input (CLI arg > config > default path)
    input_path = config.paths.pdf_path

    try:
        pdf_files = _resolve_pdf_inputs(input_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"✗ {exc}")
        logger.error("\nUsage:")
        logger.error("  python ingest.py /app/data/file.pdf")
        logger.error("  python ingest.py /app/data  # Ingest all PDFs in directory")
        logger.error("\nConfiguration:")
        logger.error("  - Config file: config.yaml or config.yml")
        logger.error("  - Or set RAG_CONFIG_FILE=/path/to/config.yaml")
        sys.exit(EXIT_CODE_ERROR)

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("RAG Document Ingestion Pipeline")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Inputs: {len(pdf_files)} PDF(s)")
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

        for pdf_file in pdf_files:
            ingest_pdf(pdf_file, vector_store, embedding_model, config)

        logger.info("✓ All PDFs processed successfully.")

    except Exception as exc:
        logger.error(f"\n✗ Error during ingestion: {exc}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()

