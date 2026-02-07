"""
Tests for ChromaDB vector database manager.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVectorDatabaseManagerInitialization:
    """Tests for VectorDatabaseManager initialization."""

    def test_should_create_manager_with_config(self) -> None:
        """Should create manager with configuration from AppConfig."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            
            assert manager is not None

    def test_should_create_persistent_client(self) -> None:
        """Should use persistent client with configured directory."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager(persist_directory=Path("./test/chroma"))
            
            mock_chroma.PersistentClient.assert_called()


class TestVectorDatabaseManagerCollection:
    """Tests for collection management."""

    def test_should_get_or_create_collection(self) -> None:
        """Should get existing collection or create new one."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            collection = manager.get_collection("test_collection")
            
            mock_client.get_or_create_collection.assert_called_once()
            assert collection is not None


class TestVectorDatabaseManagerDocuments:
    """Tests for document operations."""

    def test_should_add_documents_with_embeddings(self) -> None:
        """Should add documents and generate embeddings."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            
            manager.add_documents(
                collection_name="test",
                documents=["Document 1", "Document 2"],
                ids=["doc1", "doc2"],
                metadatas=[{"source": "test"}, {"source": "test"}]
            )
            
            mock_collection.add.assert_called_once()

    def test_should_query_similar_documents(self) -> None:
        """Should query for similar documents."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "ids": [["doc1", "doc2"]],
                "documents": [["Document 1", "Document 2"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[{"source": "test"}, {"source": "test"}]]
            }
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            
            results = manager.query_similar(
                collection_name="test",
                query_text="test query",
                n_results=2
            )
            
            assert len(results) == 2


class TestVectorDatabaseManagerHealth:
    """Tests for health checking."""

    def test_should_return_true_when_healthy(self) -> None:
        """Should return True when database is accessible."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.heartbeat.return_value = True
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            
            assert manager.is_healthy() is True

    def test_should_return_false_when_unhealthy(self) -> None:
        """Should return False when database is not accessible."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.heartbeat.side_effect = Exception("Connection failed")
            
            from src.database.vector_db import VectorDatabaseManager
            
            manager = VectorDatabaseManager()
            
            assert manager.is_healthy() is False


class TestVectorDatabaseManagerEmbeddings:
    """Tests for embedding function."""

    def test_should_use_ollama_embeddings(self) -> None:
        """Should configure Ollama embedding function."""
        with patch("src.database.vector_db.chromadb") as mock_chroma:
            with patch("src.database.vector_db.OllamaEmbeddings") as mock_ollama:
                mock_client = MagicMock()
                mock_chroma.PersistentClient.return_value = mock_client
                mock_embedding = MagicMock()
                mock_ollama.return_value = mock_embedding
                
                from src.database.vector_db import VectorDatabaseManager
                
                manager = VectorDatabaseManager(embedding_model="nomic-embed-text")
                
                # Embedding should be created with correct model
                mock_ollama.assert_called()
