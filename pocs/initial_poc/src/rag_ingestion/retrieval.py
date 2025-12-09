"""Retrieval utilities for RAG queries."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable

from langchain.schema import Document


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategies. Easily extensible."""
    
    _registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new retrieval strategy factory.
        
        Parameters
        ----------
        name
            Name to register the strategy under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Any]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> Any:
        """Create a retrieval strategy by name.
        
        Parameters
        ----------
        strategy_name
            Name of the strategy to create.
        **kwargs
            Additional arguments passed to the strategy factory.
            
        Returns
        -------
        Retrieval strategy instance.
        """
        if strategy_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown retrieval strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
        return cls._registry[strategy_name](kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._registry.keys())


# Register retrieval strategies
@RetrievalStrategyFactory.register("similarity")
def _create_similarity_strategy(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create similarity search strategy (default)."""
    return {
        "type": "similarity",
        "k": config.get("k", 4),
    }




class DocumentRetriever:
    """Retrieve relevant documents from vector store. Strategy-agnostic interface."""

    def __init__(
        self,
        vector_store: Any,
        strategy: str = "similarity",
        strategy_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the retriever.

        Parameters
        ----------
        vector_store
            Vector store instance (from VectorStoreIngester.vector_store).
        strategy
            Retrieval strategy to use.
            Use RetrievalStrategyFactory.list_strategies() to see available strategies.
            Default: "similarity"
        strategy_config
            Optional configuration for the retrieval strategy.
            For similarity: {"k": 4}
        """
        self.vector_store = vector_store
        self.strategy_name = strategy
        config = strategy_config or {}
        self.strategy = RetrievalStrategyFactory.create(strategy, **config)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.

        Parameters
        ----------
        query
            The search query.

        Returns
        -------
        List of Document objects, most relevant first.
        """
        k = self.strategy["k"]
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """Retrieve documents with similarity scores.

        Parameters
        ----------
        query
            The search query.

        Returns
        -------
        List of tuples: (Document, score), most relevant first.
        """
        k = self.strategy["k"]
        return self.vector_store.similarity_search_with_score(query, k=k)



__all__ = ["DocumentRetriever"]

