"""Protocol for embedding model implementations."""
from __future__ import annotations

from typing import Protocol, List

class Embeddings(Protocol):
    """Protocol for embedding model implementations.
    
    Compatible with LangChain's Embeddings interface and any custom
    implementation that provides these methods.
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Parameters
        ----------
        texts
            List of text strings to embed.
            
        Returns
        -------
        List of embedding vectors, each as a list of floats.
        """
        ...
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Parameters
        ----------
        text
            Query text to embed.
            
        Returns
        -------
        Embedding vector as a list of floats.
        """
        ...

