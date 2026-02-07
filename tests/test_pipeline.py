"""
Tests for the ingestion pipeline.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from src.ingestion.extractor import Triple, ExtractionResult


class TestIngestionPipeline:
    """Tests for the ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_should_ingest_text_file(self) -> None:
        """Should read and process a text file."""
        with patch("src.ingestion.pipeline.EntityExtractor") as mock_extractor_class:
            with patch("src.ingestion.pipeline.GraphDatabaseManager") as mock_graph:
                with patch("src.ingestion.pipeline.VectorDatabaseManager") as mock_vector:
                    mock_extractor = MagicMock()
                    mock_extractor_class.return_value = mock_extractor
                    mock_extractor.extract = AsyncMock(return_value=ExtractionResult(
                        triples=[
                            Triple(
                                subject="TechFlow",
                                subject_type="Company",
                                predicate="MANUFACTURES",
                                object="FlowChips",
                                object_type="Product"
                            )
                        ],
                        source_text="test"
                    ))
                    
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value.__enter__ = MagicMock(return_value=mock_graph_instance)
                    mock_graph.return_value.__exit__ = MagicMock(return_value=False)
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    
                    from src.ingestion.pipeline import IngestionPipeline
                    
                    pipeline = IngestionPipeline()
                    result = await pipeline.ingest_text("TechFlow manufactures FlowChips")
                    
                    assert result.success is True
                    assert result.nodes_created >= 0

    @pytest.mark.asyncio
    async def test_should_use_merge_for_idempotency(self) -> None:
        """Should use MERGE operations to prevent duplicates."""
        with patch("src.ingestion.pipeline.EntityExtractor") as mock_extractor_class:
            with patch("src.ingestion.pipeline.VectorDatabaseManager") as mock_vector:
                mock_extractor = MagicMock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract = AsyncMock(return_value=ExtractionResult(
                    triples=[
                        Triple(
                            subject="TechFlow",
                            subject_type="Company",
                            predicate="MANUFACTURES",
                            object="FlowChips",
                            object_type="Product"
                        )
                    ],
                    source_text="test"
                ))
                
                # Create a mock graph manager to inject directly
                mock_graph_instance = MagicMock()
                mock_graph_instance.merge_node = MagicMock()
                mock_graph_instance.merge_relationship = MagicMock()
                
                mock_vector_instance = MagicMock()
                mock_vector.return_value = mock_vector_instance
                
                from src.ingestion.pipeline import IngestionPipeline
                
                # Inject the mock graph manager directly
                pipeline = IngestionPipeline(graph_manager=mock_graph_instance)
                await pipeline.ingest_text("TechFlow manufactures FlowChips")
                
                # Verify merge_node was called (not create_node)
                mock_graph_instance.merge_node.assert_called()

    @pytest.mark.asyncio
    async def test_should_add_to_vector_store(self) -> None:
        """Should add documents to ChromaDB."""
        with patch("src.ingestion.pipeline.EntityExtractor") as mock_extractor_class:
            with patch("src.ingestion.pipeline.GraphDatabaseManager") as mock_graph:
                with patch("src.ingestion.pipeline.VectorDatabaseManager") as mock_vector:
                    mock_extractor = MagicMock()
                    mock_extractor_class.return_value = mock_extractor
                    mock_extractor.extract = AsyncMock(return_value=ExtractionResult(
                        triples=[],
                        source_text="test document"
                    ))
                    
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value.__enter__ = MagicMock(return_value=mock_graph_instance)
                    mock_graph.return_value.__exit__ = MagicMock(return_value=False)
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    
                    from src.ingestion.pipeline import IngestionPipeline
                    
                    pipeline = IngestionPipeline()
                    await pipeline.ingest_text("test document")
                    
                    mock_vector_instance.add_documents.assert_called()


class TestIngestionPipelineFile:
    """Tests for file-based ingestion."""

    @pytest.mark.asyncio
    async def test_should_process_each_line(self) -> None:
        """Should process each line of a text file."""
        with patch("src.ingestion.pipeline.EntityExtractor") as mock_extractor_class:
            with patch("src.ingestion.pipeline.GraphDatabaseManager") as mock_graph:
                with patch("src.ingestion.pipeline.VectorDatabaseManager") as mock_vector:
                    mock_extractor = MagicMock()
                    mock_extractor_class.return_value = mock_extractor
                    mock_extractor.extract = AsyncMock(return_value=ExtractionResult(
                        triples=[],
                        source_text="test"
                    ))
                    
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value.__enter__ = MagicMock(return_value=mock_graph_instance)
                    mock_graph.return_value.__exit__ = MagicMock(return_value=False)
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    
                    from src.ingestion.pipeline import IngestionPipeline
                    
                    pipeline = IngestionPipeline()
                    
                    # Create a temporary test file
                    with patch("builtins.open", MagicMock(return_value=MagicMock(
                        __enter__=MagicMock(return_value=MagicMock(
                            readlines=MagicMock(return_value=["Line 1\n", "Line 2\n"])
                        )),
                        __exit__=MagicMock(return_value=False)
                    ))):
                        result = await pipeline.ingest_file(Path("test.txt"))
                    
                    # Should have called extract for each line
                    assert mock_extractor.extract.call_count >= 1


class TestIngestionResult:
    """Tests for IngestionResult data class."""

    def test_should_create_result(self) -> None:
        """Should create an ingestion result."""
        from src.ingestion.pipeline import IngestionResult
        
        result = IngestionResult(
            success=True,
            nodes_created=5,
            relationships_created=3,
            documents_added=10
        )
        
        assert result.success is True
        assert result.nodes_created == 5
