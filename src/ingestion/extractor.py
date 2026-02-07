"""
Entity and Relationship Extraction for Atlas-GRAG.

Uses Ollama LLM to extract structured triples (subject, predicate, object)
from raw text for knowledge graph population.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM

from src.config import get_config

logger = logging.getLogger(__name__)


# Common company suffixes to normalize
COMPANY_SUFFIXES = [
    r"\s+inc\.?$",
    r"\s+incorporated$",
    r"\s+corp\.?$",
    r"\s+corporation$",
    r"\s+ltd\.?$",
    r"\s+limited$",
    r"\s+llc\.?$",
    r"\s+co\.?$",
    r"\s+company$",
    r"\s+plc\.?$",
    r"\s+gmbh$",
    r"\s+ag$",
]


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity names for consistency.
    
    Performs:
    - Lowercase conversion
    - Whitespace trimming and collapsing
    - Common suffix removal (Inc., Corp., Ltd., etc.)
    
    Args:
        name: Raw entity name
        
    Returns:
        Normalized entity name
    """
    if not name:
        return ""
    
    # Strip and lowercase
    normalized = name.strip().lower()
    
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    
    # Remove common company suffixes
    for suffix_pattern in COMPANY_SUFFIXES:
        normalized = re.sub(suffix_pattern, "", normalized, flags=re.IGNORECASE)
    
    return normalized.strip()


@dataclass
class Triple:
    """
    Represents a subject-predicate-object triple for the knowledge graph.
    
    Attributes:
        subject: The source entity (e.g., "TechFlow Inc")
        subject_type: Node label for subject (e.g., "Company")
        predicate: Relationship type (e.g., "MANUFACTURES")
        object: The target entity (e.g., "FlowChips")
        object_type: Node label for object (e.g., "Product")
        properties: Optional relationship properties
    """
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
            "properties": self.properties,
        }


@dataclass
class ExtractionResult:
    """
    Result of entity extraction from text.
    
    Attributes:
        triples: List of extracted triples
        source_text: Original input text
        error: Error message if extraction failed
    """
    triples: List[Triple] = field(default_factory=list)
    source_text: str = ""
    error: Optional[str] = None


# Knowledge graph schema for the extraction prompt
SCHEMA_DESCRIPTION = """
## Node Types
- Company: Organizations in the supply chain (name, industry, location)
- Product: Manufactured products or components (name, category)
- Location: Geographical locations (name, type, country)
- LogisticsNode: Ports, warehouses, distribution centers (name, type, capacity)
- RiskEvent: Disruption events (name, type, severity, date)

## Relationship Types
- MANUFACTURES: (Company)-[:MANUFACTURES]->(Product)
- DEPENDS_ON: (Company)-[:DEPENDS_ON]->(Company) - supply chain dependency
- STORED_IN: (Product)-[:STORED_IN]->(Location)
- COMPONENT_OF: (Product)-[:COMPONENT_OF]->(Product)
- AFFECTS: (RiskEvent)-[:AFFECTS]->(Location or Company)
- OPERATES_AT: (Company)-[:OPERATES_AT]->(Location)
- LOCATED_IN: (LogisticsNode)-[:LOCATED_IN]->(Location)
- SHIPS_VIA: (Company)-[:SHIPS_VIA]->(LogisticsNode)
- COMPETES_WITH: (Company)-[:COMPETES_WITH]->(Company)
"""


EXTRACTION_PROMPT_TEMPLATE = """You are a knowledge extraction system for supply chain analysis.
Extract entities and relationships from the given text according to the schema below.

{schema}

## Instructions
1. Identify all entities mentioned in the text
2. Determine the correct node type for each entity
3. Identify relationships between entities
4. Return ONLY valid JSON in the exact format specified

## Output Format
Return a JSON object with a "triples" array:
```json
{{
    "triples": [
        {{
            "subject": "Entity Name",
            "subject_type": "NodeType",
            "predicate": "RELATIONSHIP_TYPE",
            "object": "Another Entity",
            "object_type": "NodeType"
        }}
    ]
}}
```

## Text to Analyze
{text}

## Extracted Knowledge (JSON only, no other text):"""


class EntityExtractor:
    """
    Extracts entities and relationships from text using Ollama LLM.
    
    Uses a prompt-based approach to extract structured triples
    that can be directly loaded into the Neo4j knowledge graph.
    
    Example:
        extractor = EntityExtractor()
        result = await extractor.extract(
            "TechFlow Inc. manufactures FlowChips in Singapore."
        )
        for triple in result.triples:
            print(f"{triple.subject} -{triple.predicate}-> {triple.object}")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        normalize: bool = True,
        temperature: float = 0.0
    ) -> None:
        """
        Initialize the entity extractor.
        
        Args:
            model: Ollama model name (defaults to config)
            base_url: Ollama server URL (defaults to config)
            normalize: Whether to normalize entity names
            temperature: LLM temperature (0 for deterministic)
        """
        config = get_config().ollama
        
        self._model = model or config.model
        self._base_url = base_url or config.base_url
        self._normalize = normalize
        
        self._llm = OllamaLLM(
            model=self._model,
            base_url=self._base_url,
            temperature=temperature
        )
    
    def _build_prompt(self, text: str) -> str:
        """
        Build the extraction prompt with schema and text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Complete prompt string
        """
        return EXTRACTION_PROMPT_TEMPLATE.format(
            schema=SCHEMA_DESCRIPTION,
            text=text
        )
    
    def _parse_response(self, response: str) -> List[Triple]:
        """
        Parse LLM response into Triple objects.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            List of parsed Triple objects
            
        Raises:
            ValueError: If response cannot be parsed
        """
        # Try to extract JSON from response
        # Handle cases where LLM adds extra text
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError(f"No JSON object found in response: {response[:200]}")
        
        json_str = json_match.group()
        data = json.loads(json_str)
        
        triples = []
        for item in data.get("triples", []):
            triple = Triple(
                subject=item.get("subject", ""),
                subject_type=item.get("subject_type", "Entity"),
                predicate=item.get("predicate", "RELATED_TO"),
                object=item.get("object", ""),
                object_type=item.get("object_type", "Entity"),
                properties=item.get("properties", {})
            )
            
            # Normalize if enabled
            if self._normalize:
                triple = Triple(
                    subject=normalize_entity_name(triple.subject) or triple.subject,
                    subject_type=triple.subject_type,
                    predicate=triple.predicate.upper().replace(" ", "_"),
                    object=normalize_entity_name(triple.object) or triple.object,
                    object_type=triple.object_type,
                    properties=triple.properties
                )
            
            if triple.subject and triple.object:
                triples.append(triple)
        
        return triples
    
    async def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ExtractionResult containing triples or error
        """
        result = ExtractionResult(source_text=text)
        
        try:
            prompt = self._build_prompt(text)
            response = self._llm.invoke(prompt)
            
            triples = self._parse_response(response)
            result.triples = triples
            
            logger.info(f"Extracted {len(triples)} triples from text")
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse LLM response as JSON: {e}"
            logger.error(error_msg)
            result.error = error_msg
            
        except ValueError as e:
            error_msg = str(e)
            logger.error(error_msg)
            result.error = error_msg
            
        except Exception as e:
            error_msg = f"Extraction failed: {e}"
            logger.error(error_msg)
            result.error = error_msg
        
        return result
    
    def extract_sync(self, text: str) -> ExtractionResult:
        """
        Synchronous version of extract for non-async contexts.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ExtractionResult containing triples or error
        """
        import asyncio
        return asyncio.run(self.extract(text))
