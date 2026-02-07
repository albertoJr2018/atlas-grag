"""
Ingestion module for Atlas-GRAG.
Handles PDF parsing, entity extraction, and graph population.
"""

from src.ingestion.extractor import (
    EntityExtractor,
    ExtractionResult,
    Triple,
    normalize_entity_name,
)
from src.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "EntityExtractor",
    "ExtractionResult",
    "Triple",
    "normalize_entity_name",
    "IngestionPipeline",
    "IngestionResult",
]
