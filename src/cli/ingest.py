#!/usr/bin/env python3
"""Main script to ingest PDF documents into the vector database."""

import sys
import os
from pathlib import Path
from typing import List
from rag_ingestion import (
    PDFMarkdownLoader,
    MarkdownChunker,
    ChunkEmbedder,
    VectorStoreIngester,
)


# Default locations (can be overridden via env vars or CLI)
DEFAULT_PDF_LOCATION = Path(os.environ.get("PDF_PATH", "/app/data"))
MARKDOWN_OUTPUT_DIR = Path(os.environ.get("MARKDOWN_DIR", "/app/data/markdown"))
VECTOR_DB_PATH = Path(os.environ.get("VECTOR_DB_PATH", "/app/vector_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_collection")

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKING_METHOD = "recursive"  # "recursive", "character", or "token"

# Embedding configuration
EMBEDDING_MODEL = "huggingface"  # "huggingface" (free)

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


def ingest_pdf(pdf_path: Path, vector_store: VectorStoreIngester, embedder: ChunkEmbedder) -> None:
    print("=" * 60)
    print(f"Ingesting: {pdf_path}")
    print("=" * 60)

    # Step 1: PDF → Markdown
    print("Step 1: Converting PDF to Markdown...")
    loader = PDFMarkdownLoader(
        pdf_path,
        output_dir=_prepare_output_dir(MARKDOWN_OUTPUT_DIR),
    )
    markdown_path = loader.to_markdown_file()
    print(f"✓ Markdown saved to: {markdown_path}")

    # Step 2: Markdown → Chunks
    print(f"\nStep 2: Chunking markdown (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunker = MarkdownChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        method=CHUNKING_METHOD,
    )
    chunks = chunker.chunk_markdown_file(str(markdown_path))
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Chunks → Embeddings
    print(f"\nStep 3: Embedding chunks (model={EMBEDDING_MODEL})...")
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"✓ Embedded {len(embedded_chunks)} chunks")
    print(f"  Embedding dimension: {embedder.get_embedding_dimension()}")

    # Step 4: Embeddings → Vector Database
    print(f"\nStep 4: Storing in vector database...")
    print(f"  Database location: {VECTOR_DB_PATH}")
    print(f"  Collection: {COLLECTION_NAME}")

    vector_store.ingest_chunks(embedded_chunks)
    vector_store.persist()
    print(f"✓ Ingested {len(embedded_chunks)} chunks")

    # Summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"✓ PDF processed: {pdf_path.name}")
    print(f"✓ Markdown file: {markdown_path}")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Database location: {VECTOR_DB_PATH}")
    print(f"✓ Collection: {COLLECTION_NAME}")
    print("=" * 60 + "\n")


def main():
    """Run the ingestion pipeline for one or more PDFs."""
    # Determine input (CLI arg > env var > default path)
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = DEFAULT_PDF_LOCATION

    try:
        pdf_files = _resolve_pdf_inputs(input_path)
    except Exception as exc:
        print(f"✗ {exc}")
        print("\nUsage:")
        print("  python ingest.py /app/data/file.pdf")
        print("  python ingest.py /app/data  # Ingest all PDFs in directory")
        print("\nConfigure defaults with env vars: PDF_PATH, MARKDOWN_DIR, VECTOR_DB_PATH, COLLECTION_NAME")
        sys.exit(1)

    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)
    print(f"Inputs: {len(pdf_files)} PDF(s)")
    print(f"Database: {VECTOR_DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    try:
        embedder = ChunkEmbedder(model_name=EMBEDDING_MODEL)
        vector_store = VectorStoreIngester(
            store_name="chromadb",
            store_config={
                "persist_directory": str(VECTOR_DB_PATH),
                "collection_name": COLLECTION_NAME,
            },
            embedding_function=embedder.embedding_model,
        )

        for pdf_file in pdf_files:
            ingest_pdf(pdf_file, vector_store, embedder)

        print("✓ All PDFs processed successfully.")

    except Exception as exc:
        print(f"\n✗ Error during ingestion: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

