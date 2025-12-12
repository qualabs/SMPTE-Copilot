#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import sys
import os
import argparse
import json
from pathlib import Path
from rag_ingestion import ChunkEmbedder, VectorStoreIngester, RetrievalPipeline

# Configuration - Uses environment variables (can be overridden)
# IMPORTANT: collection_name must match COLLECTION_NAME in ingest.py
chroma_db_path = Path(os.environ.get("CHROMA_DB_PATH", "/app/chroma_db"))
collection_name = os.environ.get("COLLECTION_NAME", "rag_collection")


def load_role_mapping(mapping_file: str = "/app/role_mapping.json") -> dict:
    """Load role-to-tags mapping from JSON file.
    
    Parameters
    ----------
    mapping_file : str
        Path to the JSON file containing role-to-tags mapping.
    
    Returns
    -------
    dict
        Role-to-tags mapping, or empty dict if file doesn't exist.
    """
    try:
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load role mapping: {e}")
    return {}


def build_access_filter(user_role: str = None, user_tags: list = None, 
                       role_mapping: dict = None) -> dict:
    """Build ChromaDB metadata filter for role-aware access control.
    
    Implements the hybrid approach:
    (doc.required_role_strict == user_role) OR 
    (doc.access_tags ∩ user_authorized_tags ≠ ∅)
    
    Parameters
    ----------
    user_role : str, optional
        User's primary role.
    user_tags : list, optional
        User's direct access tags.
    role_mapping : dict, optional
        Mapping of roles to authorized tags.
    
    Returns
    -------
    dict or None
        ChromaDB filter dictionary, or None if no filtering needed.
    """
    if not user_role and not user_tags:
        return None  # No filtering (admin/testing mode)
    
    # Aggregate all authorized tags
    authorized_tags = set(user_tags or [])
    if user_role and role_mapping:
        role_tags = role_mapping.get(user_role, [])
        authorized_tags.update(role_tags)
    
    # Build ChromaDB filter
    filters = []
    
    # Add role match condition
    if user_role:
        filters.append({"required_role_strict": {"$eq": user_role}})
    
    # Add tag intersection condition
    if authorized_tags:
        # ChromaDB uses $in operator for array intersection
        filters.append({"access_tags": {"$in": list(authorized_tags)}})
    
    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        # Combine with OR logic
        return {"$or": filters}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Query vector database with optional role-aware access control"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query string"
    )
    parser.add_argument(
        "--role",
        type=str,
        default="",
        help="User's role for access control (e.g., 'Finance_Manager')"
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="User's direct access tags, comma-separated (e.g., 'Finance,Public')"
    )
    parser.add_argument(
        "--role-mapping",
        type=str,
        default="/app/role_mapping.json",
        help="Path to role-to-tags mapping JSON file"
    )
    
    args = parser.parse_args()
    
    # Parse user permissions
    query = args.query
    user_role = args.role.strip() or None
    user_tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    
    # Load role mapping
    role_mapping = load_role_mapping(args.role_mapping)
    
    # Build access filter
    metadata_filter = build_access_filter(user_role, user_tags, role_mapping)
    
    print("=" * 60)
    print("Querying Vector Database")
    print("=" * 60)
    print(f"Query: {query}")
    if metadata_filter:
        print(f"User Role: {user_role or 'None'}")
        print(f"User Tags: {user_tags or 'None'}")
        if user_role and role_mapping and user_role in role_mapping:
            print(f"Role Mapped Tags: {role_mapping[user_role]}")
        print(f"Applied Filter: {metadata_filter}")
    else:
        print("⚠️  No access filter applied (admin/testing mode)")
    print()
    
    try:
        # Step 1: Check if database exists
        chroma_db_path_obj = Path(chroma_db_path)
        if not chroma_db_path_obj.exists():
            print(f"✗ Error: Vector database not found: {chroma_db_path_obj}")
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
                "persist_directory": str(chroma_db_path_obj),
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
        
        # Step 5: Query the database using pipeline (with scores and filter)
        print("Searching...")
        results_with_scores = pipeline.retrieve_with_scores(query, metadata_filter=metadata_filter)
        
        # Step 6: Display results with similarity scores
        if len(results_with_scores) == 0:
            print("\n⚠️  No documents found matching your query and access permissions.")
            print("    This could mean:")
            print("    - No documents match the search query")
            print("    - You don't have access to documents containing relevant information")
            print("    - No documents have been ingested yet")
        else:
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
                    # Display access control metadata if present
                    access_info = {}
                    if "access_tags" in doc.metadata:
                        access_info["access_tags"] = doc.metadata["access_tags"]
                    if "required_role_strict" in doc.metadata:
                        access_info["required_role_strict"] = doc.metadata["required_role_strict"]
                    print(f"    Metadata: {doc.metadata}")
                    if access_info:
                        print(f"    Access Control: {access_info}")
        
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

