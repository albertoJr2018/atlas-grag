"""
ChromaDB Vector Database Manager for Atlas-GRAG.

Handles vector embeddings storage and semantic similarity search
using Ollama embeddings for local-first operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.config import get_config

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Custom embedding function using Ollama for local embeddings.
    
    Compatible with ChromaDB's embedding function interface.
    """
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ) -> None:
        """
        Initialize Ollama embeddings.
        
        Args:
            model: Ollama embedding model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self._client: Optional[Any] = None
    
    @property
    def client(self) -> Any:
        """Lazy-load the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("ollama package is required for embeddings")
        return self._client
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Args:
            input: List of strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input:
            try:
                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Default dimension
        return embeddings


class VectorDatabaseManager:
    """
    Manager for ChromaDB vector database operations.
    
    Provides document storage, embedding generation via Ollama,
    and semantic similarity search.
    
    Usage:
        manager = VectorDatabaseManager()
        manager.add_documents(
            collection_name="supply_chain",
            documents=["doc1", "doc2"],
            ids=["id1", "id2"]
        )
        results = manager.query_similar("supply chain risk", n_results=5)
    """
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the vector database manager.
        
        Args:
            persist_directory: Directory for persistent storage
            embedding_model: Ollama model for embeddings
            ollama_base_url: Ollama server URL
        """
        config = get_config()
        
        self._persist_dir = persist_directory or config.chroma.persist_directory
        self._embedding_model = embedding_model or config.ollama.embedding_model
        self._ollama_url = ollama_base_url or config.ollama.base_url
        
        # Ensure persist directory exists
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding function
        self._embedding_fn = OllamaEmbeddings(
            model=self._embedding_model,
            base_url=self._ollama_url
        )
        
        # Cache for collections
        self._collections: Dict[str, Any] = {}
    
    def is_healthy(self) -> bool:
        """
        Check if the database is healthy.
        
        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
    
    def get_collection(self, name: str) -> Any:
        """
        Get or create a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection
        """
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_fn
            )
        return self._collections[name]
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection name
            documents: List of document texts
            ids: List of unique document IDs
            metadatas: Optional list of metadata dictionaries
        """
        collection = self.get_collection(collection_name)
        
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    def query_similar(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar documents.
        
        Args:
            collection_name: Collection to search
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of similar documents with metadata and distances
        """
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        # Transform results into list of dicts
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = {
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None
                }
                documents.append(doc)
        
        return documents
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
        """
        self._client.delete_collection(name)
        if name in self._collections:
            del self._collections[name]
    
    def get_document_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Document count
        """
        collection = self.get_collection(collection_name)
        return collection.count()
