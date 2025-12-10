#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import sys
from pathlib import Path
from src import EmbeddingModelFactory, VectorStoreFactory, RetrieverFactory, Config

def main():
    config = Config.get_config()
    
    # Get query from command line argument
    if len(sys.argv) < 2:
        print("Usage: python query.py 'your question here'")
        print("\nExample:")
        print("  python query.py 'What is the main topic?'")
        print("\nConfiguration:")
        print("  - Config file: config.yaml or config.yml")
        print("  - Or set RAG_CONFIG_FILE=/path/to/config.yaml")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("Querying Vector Database")
    print("=" * 60)
    print(f"Query: {query}\n")
    
    try:
        # Step 1: Check if database exists
        vector_db_path = config.vector_store.persist_directory
        if not vector_db_path.exists():
            print(f"✗ Error: Vector database not found: {vector_db_path}")
            print("\nPlease ingest documents first:")
            print("  python ingest.py /path/to/document.pdf")
            print("\nOr set RAG_CONFIG_FILE or environment variables, then run:")
            print("  python ingest.py")
            sys.exit(1)
        
        # Step 2: Create embedding model using factory
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.model_name,
            **(config.embedding.model_config or {}),
        )
        
        # Step 3: Create vector store using factory
        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            persist_directory=str(vector_db_path),
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_model,
            **(config.vector_store.store_config or {}),
        )
        
        # Step 4: Create retriever
        searcher_config = {"k": config.retrieval.k}
        if config.retrieval.searcher_config:
            searcher_config.update(config.retrieval.searcher_config)
        searcher_config["vector_store"] = vector_store
        
        retriever = RetrieverFactory.create(
            config.retrieval.searcher_strategy,
            **searcher_config,
        )
        
        # Step 5: Query the database using retriever (with scores)
        print("Searching...")
        results_with_scores = retriever.retrieve_with_scores(query)
        
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
