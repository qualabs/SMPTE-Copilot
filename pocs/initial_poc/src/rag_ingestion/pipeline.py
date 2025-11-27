"""Retrieval pipeline for RAG queries."""
from __future__ import annotations

from typing import List, Optional, Any

from langchain.schema import Document

from .embeddings import ChunkEmbedder
from .retrieval import DocumentRetriever


class RetrievalPipeline:
    """Basic retrieval pipeline for RAG queries.
    
    Pipeline flow:
    1. Query → Embed (using ChunkEmbedder)
    2. Embedding → Search (using DocumentRetriever)
    3. Return documents
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any = None,
        embedder: Optional[ChunkEmbedder] = None,
        searcher_strategy: str = "similarity",
        searcher_config: Optional[dict] = None,
    ):
        """Initialize the retrieval pipeline.

        Parameters
        ----------
        vector_store
            Vector store instance (from VectorStoreIngester.vector_store).
        embedding_model
            Embedding model to use for query embedding.
            If None, will create a default ChunkEmbedder.
        embedder
            Pre-initialized ChunkEmbedder instance.
            If provided, embedding_model is ignored.
        searcher_strategy
            Retrieval strategy for DocumentRetriever.
            Default: "similarity"
        searcher_config
            Configuration for the searcher strategy.
            For similarity: {"k": 4}
        """
        self.vector_store = vector_store
        
        # Initialize embedder
        if embedder is not None:
            self.embedder = embedder
        elif embedding_model is not None:
            self.embedder = ChunkEmbedder(embedding_model=embedding_model)
        else:
            # Default to HuggingFace (free)
            self.embedder = ChunkEmbedder(model_name="huggingface")
        
        # Initialize searcher
        config = searcher_config or {}
        self.retriever = DocumentRetriever(
            vector_store=vector_store,
            strategy=searcher_strategy,
            strategy_config=config,
        )
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

    def retrieve_with_scores(self, query: str) -> List[tuple]:
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

