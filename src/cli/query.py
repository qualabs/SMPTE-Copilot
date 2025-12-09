#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import sys
import os
from pathlib import Path
from rag_ingestion import ChunkEmbedder, VectorStoreIngester, RetrievalPipeline

# Configuration - Uses environment variables (can be overridden)
# IMPORTANT: collection_name must match COLLECTION_NAME in ingest.py
vector_db_path = Path(os.environ.get("VECTOR_DB_PATH", "/app/vector_db"))
collection_name = os.environ.get("COLLECTION_NAME", "rag_collection")

def main():
    # Get query from command line argument
    if len(sys.argv) < 2:
        print("Usage: python query.py 'your question here'")
        print("\nExample:")
        print("  python query.py 'What is the main topic?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("Querying Vector Database")
    print("=" * 60)
    print(f"Query: {query}\n")
    
    try:
        # Step 1: Check if database exists
        vector_db_path_obj = Path(vector_db_path)
        if not vector_db_path_obj.exists():
            print(f"✗ Error: Vector database not found: {vector_db_path_obj}")
            print("\nPlease ingest documents first:")
            print("  python ingest.py /path/to/document.pdf")
            print("\nOr edit ingest.py to set PDF_PATH, then run:")
            print("  python ingest.py")
            sys.exit(1)
        
        # Step 2: Initialize embedder
        embedder = ChunkEmbedder(model_name="huggingface")
        
        # Step 3: Load vector store
        vector_store = VectorStoreIngester(
            store_name="chromadb",
            store_config={
                "persist_directory": str(vector_db_path_obj),
                "collection_name": collection_name,
            },
            embedding_function=embedder.embedding_model,
        )
        
        # Step 4: Create retrieval pipeline
        pipeline = RetrievalPipeline(
            vector_store=vector_store.vector_store,
            embedder=embedder,
            searcher_strategy="similarity",
            searcher_config={"k": 5},  # Return top 5 results
        )
        
        # Step 5: Query the database using pipeline (with scores)
        print("Searching...")
        results_with_scores = pipeline.retrieve_with_scores(query)
        
        # Step 6: Display results with similarity scores
        print(f"\nFound {len(results_with_scores)} relevant documents:\n")
        print("-" * 60)
        print("Similarity Score Guide:")
        print("  - Higher score = More similar to query")
        print("  - Score range depends on distance metric (usually 0-1 or 0-2)")
        print("  - For cosine similarity: closer to 1 = more similar")
        print("-" * 60)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n[{i}] Similarity Score: {score:.4f}")
            print(f"    Content: {doc.page_content}")
            if doc.metadata:
                print(f"    Metadata: {doc.metadata}")
        
        print("\n" + "=" * 60)
        print("Note: Higher similarity scores indicate better matches to your query.")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

