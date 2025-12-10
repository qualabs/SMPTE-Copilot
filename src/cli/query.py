#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import sys
from pathlib import Path
from src import EmbeddingModelFactory, VectorStoreFactory, RetrieverFactory, Config
from src.cli.constants import (
    MIN_CLI_ARGS,
    SEPARATOR_LENGTH,
    SEPARATOR_CHAR,
    ALT_SEPARATOR_CHAR,
    ENUMERATE_START,
    SCORE_DECIMAL_PLACES,
    MIN_SCORE,
    MAX_SCORE_COSINE,
    MAX_SCORE_DISTANCE,
    EXIT_CODE_ERROR,
)

def main():
    config = Config.get_config()
    
    query = " ".join(sys.argv[1:])
    
    print(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    print("Querying Vector Database")
    print(SEPARATOR_CHAR * SEPARATOR_LENGTH)
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
            sys.exit(EXIT_CODE_ERROR)
        
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
        print(ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)
        print("Similarity Score Guide:")
        print("  - Higher score = More similar to query")
        print(f"  - Score range depends on distance metric (usually {MIN_SCORE}-{MAX_SCORE_COSINE} or {MIN_SCORE}-{MAX_SCORE_DISTANCE})")
        print(f"  - For cosine similarity: closer to {MAX_SCORE_COSINE} = more similar")
        print(ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)
        
        for i, (doc, score) in enumerate(results_with_scores, ENUMERATE_START):
            print(f"\n[{i}] Similarity Score: {score:.{SCORE_DECIMAL_PLACES}f}")
            print(f"    Content: {doc.page_content}")
            if doc.metadata:
                print(f"    Metadata: {doc.metadata}")
        
        print("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        print("Note: Higher similarity scores indicate better matches to your query.")
        print(SEPARATOR_CHAR * SEPARATOR_LENGTH)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(EXIT_CODE_ERROR)

if __name__ == "__main__":
    main()
