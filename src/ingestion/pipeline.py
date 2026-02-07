"""
Ingestion Pipeline for Atlas-GRAG.

Orchestrates the flow from raw text to knowledge graph population:
Text -> Entity Extraction -> Neo4j MERGE + ChromaDB Add

This module ensures idempotency through MERGE operations,
preventing duplicate nodes when running ingestion multiple times.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import get_config
from src.database.graph_db import GraphDatabaseManager
from src.database.vector_db import VectorDatabaseManager
from src.ingestion.extractor import EntityExtractor, ExtractionResult, Triple

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Result of an ingestion operation.
    
    Attributes:
        success: Whether the ingestion completed successfully
        nodes_created: Number of nodes created/merged
        relationships_created: Number of relationships created/merged
        documents_added: Number of documents added to vector store
        errors: List of error messages if any
    """
    success: bool = True
    nodes_created: int = 0
    relationships_created: int = 0
    documents_added: int = 0
    errors: List[str] = field(default_factory=list)


class IngestionPipeline:
    """
    Pipeline for ingesting text into the Atlas-GRAG knowledge base.
    
    Handles:
    - Entity and relationship extraction via LLM
    - Idempotent node/relationship creation in Neo4j
    - Document embedding in ChromaDB
    
    Example:
        pipeline = IngestionPipeline()
        result = await pipeline.ingest_file(Path("data/supply_chain.txt"))
        print(f"Created {result.nodes_created} nodes")
    """
    
    def __init__(
        self,
        graph_manager: Optional[GraphDatabaseManager] = None,
        vector_manager: Optional[VectorDatabaseManager] = None,
        extractor: Optional[EntityExtractor] = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.
        
        Args:
            graph_manager: Neo4j connection manager (created if not provided)
            vector_manager: ChromaDB manager (created if not provided)
            extractor: Entity extractor (created if not provided)
        """
        self._graph_manager = graph_manager
        self._vector_manager = vector_manager or VectorDatabaseManager()
        self._extractor = extractor or EntityExtractor()
        
        config = get_config()
        self._collection_name = config.chroma.collection_name
    
    def _generate_doc_id(self, text: str) -> str:
        """
        Generate a deterministic document ID from text content.
        
        Args:
            text: Document text
            
        Returns:
            Unique document ID based on content hash
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"doc_{text_hash}"
    
    async def _create_node(
        self,
        graph: GraphDatabaseManager,
        label: str,
        name: str
    ) -> bool:
        """
        Create a node using MERGE for idempotency.
        
        Args:
            graph: Graph database manager
            label: Node label
            name: Node name
            
        Returns:
            True if successful
        """
        try:
            graph.merge_node(
                label=label,
                properties={"name": name}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create node {label}:{name}: {e}")
            return False
    
    async def _create_relationship(
        self,
        graph: GraphDatabaseManager,
        triple: Triple
    ) -> bool:
        """
        Create a relationship using MERGE for idempotency.
        
        Args:
            graph: Graph database manager
            triple: The triple containing relationship info
            
        Returns:
            True if successful
        """
        try:
            graph.merge_relationship(
                from_label=triple.subject_type,
                from_props={"name": triple.subject},
                to_label=triple.object_type,
                to_props={"name": triple.object},
                rel_type=triple.predicate,
                rel_props=triple.properties or None
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship {triple.predicate}: {e}")
            return False
    
    async def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> IngestionResult:
        """
        Ingest a single text document.
        
        Extracts entities and relationships, adds to graph,
        and stores document in vector database.
        
        Args:
            text: Raw text to ingest
            metadata: Optional metadata for the document
            
        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult()
        metadata = metadata or {}
        
        # Step 1: Extract entities and relationships
        extraction = await self._extractor.extract(text)
        
        if extraction.error:
            result.errors.append(extraction.error)
            logger.warning(f"Extraction error: {extraction.error}")
        
        # Step 2: Add to graph database
        if extraction.triples:
            # Use context manager for graph operations
            graph = self._graph_manager or GraphDatabaseManager()
            try:
                with graph if self._graph_manager is None else self._null_context():
                    target_graph = graph if self._graph_manager is None else self._graph_manager
                    
                    for triple in extraction.triples:
                        # Create subject node
                        if await self._create_node(target_graph, triple.subject_type, triple.subject):
                            result.nodes_created += 1
                        
                        # Create object node
                        if await self._create_node(target_graph, triple.object_type, triple.object):
                            result.nodes_created += 1
                        
                        # Create relationship
                        if await self._create_relationship(target_graph, triple):
                            result.relationships_created += 1
            except Exception as e:
                result.errors.append(f"Graph database error: {e}")
                result.success = False
                logger.error(f"Graph database error: {e}")
        
        # Step 3: Add to vector database
        try:
            doc_id = self._generate_doc_id(text)
            self._vector_manager.add_documents(
                collection_name=self._collection_name,
                documents=[text],
                ids=[doc_id],
                metadatas=[{**metadata, "source": "ingestion"}]
            )
            result.documents_added = 1
        except Exception as e:
            result.errors.append(f"Vector database error: {e}")
            logger.error(f"Vector database error: {e}")
        
        return result
    
    @staticmethod
    def _null_context():
        """Null context manager for when we already have a graph manager."""
        class NullContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NullContext()
    
    async def ingest_file(
        self,
        file_path: Path,
        batch_size: int = 10
    ) -> IngestionResult:
        """
        Ingest all documents from a text file.
        
        Each line is treated as a separate document.
        
        Args:
            file_path: Path to the text file
            batch_size: Number of documents to process in parallel
            
        Returns:
            Aggregated IngestionResult
        """
        result = IngestionResult()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to read file: {e}")
            return result
        
        # Clean and filter lines
        lines = [line.strip() for line in lines if line.strip()]
        
        logger.info(f"Ingesting {len(lines)} documents from {file_path}")
        
        # Process in batches
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.ingest_text(
                    text=line,
                    metadata={"file": str(file_path), "line_number": i + j + 1}
                )
                for j, line in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    result.errors.append(str(batch_result))
                elif isinstance(batch_result, IngestionResult):
                    result.nodes_created += batch_result.nodes_created
                    result.relationships_created += batch_result.relationships_created
                    result.documents_added += batch_result.documents_added
                    result.errors.extend(batch_result.errors)
        
        logger.info(
            f"Ingestion complete: {result.nodes_created} nodes, "
            f"{result.relationships_created} relationships, "
            f"{result.documents_added} documents"
        )
        
        return result
    
    async def ingest_sample_data(self) -> IngestionResult:
        """
        Ingest the sample supply chain dataset.
        
        Convenience method to quickly populate the knowledge base
        with the default sample data.
        
        Returns:
            IngestionResult with statistics
        """
        sample_file = Path(__file__).parent.parent.parent / "data" / "sample_supply_chain.txt"
        
        if not sample_file.exists():
            return IngestionResult(
                success=False,
                errors=[f"Sample file not found: {sample_file}"]
            )
        
        return await self.ingest_file(sample_file)
