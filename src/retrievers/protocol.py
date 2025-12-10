"""Protocol for retriever implementations."""
from __future__ import annotations

from typing import Protocol, List

from langchain.schema import Document


class Retriever(Protocol):
    """Protocol for document retriever implementations.
    
    Any class implementing these methods can retrieve documents using
    different strategies (similarity search, MMR, etc.).
    """
    
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
        ...
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve documents with similarity scores.
        
        Parameters
        ----------
        query
            The search query.
            
        Returns
        -------
        List of tuples: (Document, score), most relevant first.
        """
        ...

