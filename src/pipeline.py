"""Retrieval pipeline for RAG queries."""
from __future__ import annotations

from typing import List, Optional

from langchain.schema import Document

from .embeddings import ChunkEmbedder
from .retrievers import RetrieverFactory
from .vector_stores import VectorStore


class RetrievalPipeline:
    """Basic retrieval pipeline for RAG queries.
    
    Pipeline flow:
    1. Query → Embed (using ChunkEmbedder)
    2. Embedding → Search (using DocumentRetriever)
    3. Return documents
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[ChunkEmbedder] = None,
        model_name: str = "huggingface",
        model_config: Optional[dict] = None,
        searcher_strategy: str = "similarity",
        searcher_config: Optional[dict] = None,
    ):
        """Initialize the retrieval pipeline.

        Parameters
        ----------
        vector_store
            Vector store instance (created via VectorStoreFactory).
        embedder
            Pre-initialized ChunkEmbedder instance.
            If provided, model_name and model_config are ignored.
        model_name
            Name of the embedding model to use.
            Default: "huggingface"
        model_config
            Optional configuration for the embedding model.
        searcher_strategy
            Retrieval strategy name.
            Use RetrieverFactory.list_retrievers() to see available strategies.
            Default: "similarity"
        searcher_config
            Configuration for the searcher strategy.
            For similarity: {"k": 4}
        """
        self.vector_store = vector_store
        
        # Initialize embedder
        if embedder is not None:
            self.embedder = embedder
        else:
            config = model_config or {}
            self.embedder = ChunkEmbedder(model_name=model_name, model_config=config)
        
        # Initialize searcher using factory
        config = searcher_config or {}
        config["vector_store"] = vector_store
        self.retriever = RetrieverFactory.create(searcher_strategy, **config)
        self.searcher_strategy = searcher_strategy

    def retrieve(self, query: str) -> List[Document]:
        """Run the retrieval pipeline.

        Parameters
        ----------
        query
            The search query string.

        Returns
        -------
        List of Document objects, most relevant first.
        """
        # Step 2: Embed query (query is already text, embedding happens in search)
        # The retriever handles embedding internally via the vector store's embedding function
        
        # Step 3: Search vector store
        results = self.retriever.retrieve(query)
        
        return results

    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Run the retrieval pipeline and return results with similarity scores.

        Parameters
        ----------
        query
            The search query string.

        Returns
        -------
        List of tuples: (Document, score), most relevant first.
        """
        return self.retriever.retrieve_with_scores(query)



__all__ = ["RetrievalPipeline"]

