"""
Tests for LLM-based entity and relationship extraction.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.extractor import (
    EntityExtractor,
    Triple,
    normalize_entity_name,
    ExtractionResult,
)


class TestNormalizeEntityName:
    """Tests for entity name normalization."""

    def test_should_lowercase_name(self) -> None:
        """Should convert to lowercase for consistency."""
        result = normalize_entity_name("Apple Inc")
        
        assert result == result.lower()

    def test_should_strip_whitespace(self) -> None:
        """Should remove leading/trailing whitespace."""
        result = normalize_entity_name("  TechFlow  ")
        
        assert result == "techflow"

    def test_should_normalize_common_suffixes(self) -> None:
        """Should normalize company suffixes like Inc., Corp., Ltd."""
        assert normalize_entity_name("Apple Inc.") == "apple"
        assert normalize_entity_name("Microsoft Corporation") == "microsoft"
        assert normalize_entity_name("Samsung Ltd") == "samsung"

    def test_should_handle_empty_string(self) -> None:
        """Should return empty string for empty input."""
        result = normalize_entity_name("")
        
        assert result == ""

    def test_should_remove_extra_spaces(self) -> None:
        """Should collapse multiple spaces into one."""
        result = normalize_entity_name("Tech   Flow   Inc")
        
        assert "  " not in result


class TestTriple:
    """Tests for Triple data class."""

    def test_should_create_triple(self) -> None:
        """Should create a valid triple."""
        triple = Triple(
            subject="TechFlow Inc",
            subject_type="Company",
            predicate="MANUFACTURES",
            object="FlowChips",
            object_type="Product"
        )
        
        assert triple.subject == "TechFlow Inc"
        assert triple.predicate == "MANUFACTURES"
        assert triple.object == "FlowChips"

    def test_should_have_optional_properties(self) -> None:
        """Should support optional properties dictionary."""
        triple = Triple(
            subject="TechFlow Inc",
            subject_type="Company",
            predicate="MANUFACTURES",
            object="FlowChips",
            object_type="Product",
            properties={"since": 2020, "capacity": "1M units"}
        )
        
        assert triple.properties["since"] == 2020


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.mark.asyncio
    async def test_should_extract_entities_from_text(self) -> None:
        """Should extract entities and relationships from raw text."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = """
            {
                "triples": [
                    {
                        "subject": "TechFlow Inc",
                        "subject_type": "Company",
                        "predicate": "MANUFACTURES",
                        "object": "FlowChips",
                        "object_type": "Product"
                    }
                ]
            }
            """
            
            extractor = EntityExtractor()
            result = await extractor.extract("TechFlow Inc. manufactures FlowChips")
            
            assert len(result.triples) == 1
            # Subject is normalized: lowercase, Inc suffix removed
            assert result.triples[0].subject == "techflow"

    @pytest.mark.asyncio
    async def test_should_handle_multiple_relationships(self) -> None:
        """Should extract multiple relationships from complex text."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = """
            {
                "triples": [
                    {
                        "subject": "TechFlow Inc",
                        "subject_type": "Company",
                        "predicate": "MANUFACTURES",
                        "object": "FlowChips",
                        "object_type": "Product"
                    },
                    {
                        "subject": "TechFlow Inc",
                        "subject_type": "Company",
                        "predicate": "OPERATES_AT",
                        "object": "Singapore",
                        "object_type": "Location"
                    }
                ]
            }
            """
            
            extractor = EntityExtractor()
            result = await extractor.extract(
                "TechFlow Inc. manufactures FlowChips in Singapore"
            )
            
            assert len(result.triples) == 2

    @pytest.mark.asyncio
    async def test_should_normalize_extracted_entities(self) -> None:
        """Should normalize entity names for consistency."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = """
            {
                "triples": [
                    {
                        "subject": "TechFlow Inc.",
                        "subject_type": "Company",
                        "predicate": "MANUFACTURES",
                        "object": "FlowChips",
                        "object_type": "Product"
                    }
                ]
            }
            """
            
            extractor = EntityExtractor(normalize=True)
            result = await extractor.extract("TechFlow Inc. manufactures FlowChips")
            
            # Normalized: no trailing period, lowercase
            assert "inc." not in result.triples[0].subject.lower()

    @pytest.mark.asyncio
    async def test_should_handle_llm_json_parsing_error(self) -> None:
        """Should handle malformed LLM response gracefully."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = "This is not valid JSON"
            
            extractor = EntityExtractor()
            result = await extractor.extract("Some text")
            
            assert len(result.triples) == 0
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_should_use_configured_model(self) -> None:
        """Should use the specified Ollama model."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = '{"triples": []}'
            
            extractor = EntityExtractor(model="llama3")
            await extractor.extract("Test")
            
            mock_llm_class.assert_called()


class TestExtractionPrompt:
    """Tests for the extraction prompt template."""

    def test_should_include_schema_in_prompt(self) -> None:
        """Should include knowledge graph schema in prompt."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            extractor = EntityExtractor()
            prompt = extractor._build_prompt("test text")
            
            # Should mention node types
            assert "Company" in prompt
            assert "Product" in prompt
            assert "Location" in prompt
            
            # Should mention relationship types
            assert "MANUFACTURES" in prompt
            assert "DEPENDS_ON" in prompt

    def test_should_request_json_output(self) -> None:
        """Should instruct LLM to return JSON format."""
        with patch("src.ingestion.extractor.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            extractor = EntityExtractor()
            prompt = extractor._build_prompt("test text")
            
            assert "JSON" in prompt or "json" in prompt
