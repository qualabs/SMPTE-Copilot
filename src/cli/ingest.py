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
    get_config,
)
from src.vector_stores.helpers import VectorStoreHelper
from src.embeddings.helpers import EmbeddingHelper

def _resolve_pdf_inputs(input_path: Path) -> List[Path]:
    """Resolve input path to a list of PDF files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {input_path}")
        return [input_path]
    # Directory: collect all PDFs (non-recursive)
    pdf_files = sorted(input_path.glob("*.pdf"))
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
    print("=" * 60)
    print(f"Ingesting: {pdf_path}")
    print("=" * 60)

    # Step 1: PDF → Markdown
    print("Step 1: Converting PDF to Markdown...")
    loader = LoaderFactory.create(
        "pymupdf",
        pdf_path=str(pdf_path),
        output_dir=str(_prepare_output_dir(config.paths.markdown_dir)),
    )
    markdown_path = loader.to_markdown_file()
    print(f"✓ Markdown saved to: {markdown_path}")

    # Step 2: Markdown → Chunks
    print(
        f"\nStep 2: Chunking markdown (size={config.chunking.chunk_size}, "
        f"overlap={config.chunking.chunk_overlap})..."
    )
    chunker = ChunkerFactory.create(
        "langchain",
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        method=config.chunking.method,
    )
    chunks = chunker.chunk_markdown_file(str(markdown_path))
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Chunks → Embeddings
    print(f"\nStep 3: Embedding chunks (model={config.embedding.model_name})...")
    embedded_chunks = EmbeddingHelper.embed_chunks(embedding_model, chunks, model_name=config.embedding.model_name)
    print(f"✓ Embedded {len(embedded_chunks)} chunks")

    # Step 4: Embeddings → Vector Database
    print(f"\nStep 4: Storing in vector database...")
    print(f"  Database location: {config.vector_store.persist_directory}")
    print(f"  Collection: {config.vector_store.collection_name}")

    VectorStoreHelper.ingest_chunks_with_embeddings(vector_store, embedded_chunks)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    print(f"✓ Ingested {len(embedded_chunks)} chunks")

    # Summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"✓ PDF processed: {pdf_path.name}")
    print(f"✓ Markdown file: {markdown_path}")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Database location: {config.vector_store.persist_directory}")
    print(f"✓ Collection: {config.vector_store.collection_name}")
    print("=" * 60 + "\n")


def main():
    """Run the ingestion pipeline for one or more PDFs."""
    # Load configuration
    config = get_config()
    
    # Determine input (CLI arg > config > default path)
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = config.paths.pdf_path

    try:
        pdf_files = _resolve_pdf_inputs(input_path)
    except Exception as exc:
        print(f"✗ {exc}")
        print("\nUsage:")
        print("  python ingest.py /app/data/file.pdf")
        print("  python ingest.py /app/data  # Ingest all PDFs in directory")
        print("\nConfiguration:")
        print("  - Config file: config.yaml or config.yml")
        print("  - Or set RAG_CONFIG_FILE=/path/to/config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)
    print(f"Inputs: {len(pdf_files)} PDF(s)")
    print(f"Database: {config.vector_store.persist_directory}")
    print(f"Collection: {config.vector_store.collection_name}")
    print(f"Chunk size: {config.chunking.chunk_size}, overlap: {config.chunking.chunk_overlap}")
    print(f"Embedding model: {config.embedding.model_name}")
    print()

    try:
        # Create embedding model using factory
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.model_name,
            **(config.embedding.model_config or {}),
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

        print("✓ All PDFs processed successfully.")

    except Exception as exc:
        print(f"\n✗ Error during ingestion: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

