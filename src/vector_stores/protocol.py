"""Protocol for vector store implementations."""
from __future__ import annotations

from typing import Protocol, List, Optional, Dict, Any

from langchain.schema import Document

from ..constants import DEFAULT_RETRIEVAL_K


class VectorStore(Protocol):
    """Protocol for vector store implementations.
    
    Any class implementing these methods is compatible with VectorStore,
    regardless of inheritance hierarchy. This allows LangChain's vector
    stores (Chroma, Pinecone, etc.) to work seamlessly without modification.
    
    Methods marked as optional may not be available in all implementations.
    """
    
    def similarity_search(
        self, 
        query: str, 
        k: int = DEFAULT_RETRIEVAL_K
    ) -> List[Document]:
        """Search for similar documents.
        
        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.
            
        Returns
        -------
        List of Document objects, most similar first.
        """
        ...
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = DEFAULT_RETRIEVAL_K
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.
            
        Returns
        -------
        List of tuples: (Document, score), most similar first.
        Higher scores indicate better matches.
        """
        ...
    
    def add_documents(
        self, 
        documents: List[Document]
    ) -> List[str]:
        """Add documents to the vector store.
        
        Parameters
        ----------
        documents
            List of Document objects to add.
            
        Returns
        -------
        List of document IDs (if supported).
        """
        ...
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Add texts to the vector store (optional method).
        
        Some vector stores support adding texts directly with optional
        embeddings and metadata. This method is optional but recommended
        for better performance when embeddings are pre-computed.
        
        Parameters
        ----------
        texts
            List of text strings to add.
        metadatas
            Optional list of metadata dictionaries.
        ids
            Optional list of document IDs.
        embeddings
            Optional list of pre-computed embedding vectors.
            
        Returns
        -------
        List of document IDs (if supported).
        """
        ...
    
    def persist(self) -> None:
        """Persist the vector store to disk (optional method).
        
        Not all vector stores support persistence. This method should
        be called if available to save the vector store state.
        """
        ...
    
    def delete(self, ids: Optional[List[str]] = None) -> None:
        """Delete documents or the entire collection (optional method).
        
        Parameters
        ----------
        ids
            Optional list of document IDs to delete.
            If None, may delete the entire collection (implementation-dependent).
        """
        ...

