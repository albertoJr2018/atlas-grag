"""
Tests for LLM reasoning chains.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.retriever.hybrid import RetrievalResult


class TestReasoningResponse:
    """Tests for ReasoningResponse data class."""

    def test_should_parse_structured_response(self) -> None:
        """Should parse entities, reasoning, and answer from response."""
        from src.llm.chains import ReasoningResponse
        
        response_text = """
        <entities>
        - Singapore
        - GlobalTech
        - TechFlow
        </entities>
        
        <reasoning>
        Singapore port strike affects TechFlow's operations.
        TechFlow manufactures FlowChips.
        FlowChips are used by GlobalTech.
        </reasoning>
        
        <answer>
        The Singapore strike will disrupt GlobalTech's supply chain through
        their dependency on FlowChips manufactured by TechFlow in Singapore.
        </answer>
        """
        
        parsed = ReasoningResponse.parse_from_response(response_text)
        
        assert "Singapore" in parsed.entities
        assert "FlowChips" in parsed.reasoning
        assert "disrupt" in parsed.answer.lower() or "supply chain" in parsed.answer.lower()

    def test_should_handle_unstructured_response(self) -> None:
        """Should handle response without XML tags."""
        from src.llm.chains import ReasoningResponse
        
        response_text = "The strike will impact operations significantly."
        
        parsed = ReasoningResponse.parse_from_response(response_text)
        
        assert parsed.answer == response_text
        assert parsed.entities == ""
        assert parsed.reasoning == ""


class TestReasoningChain:
    """Tests for ReasoningChain."""

    def test_should_create_with_default_config(self) -> None:
        """Should create chain with default configuration."""
        with patch("src.llm.chains.OllamaLLM") as mock_llm:
            mock_llm.return_value = MagicMock()
            
            from src.llm.chains import ReasoningChain
            
            chain = ReasoningChain()
            
            assert chain is not None

    def test_should_generate_reasoned_answer(self) -> None:
        """Should generate answer using Chain of Thought."""
        with patch("src.llm.chains.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.return_value = """
            <entities>
            - Singapore
            </entities>
            
            <reasoning>
            The strike affects shipping.
            </reasoning>
            
            <answer>
            GlobalTech will be impacted.
            </answer>
            """
            
            from src.llm.chains import ReasoningChain
            
            chain = ReasoningChain()
            
            result = RetrievalResult(
                query="test query",
                vector_chunks=["Singapore strike news"],
                graph_context="Singapore -> FlowChips"
            )
            
            response = chain.reason(result, "How will strike impact GlobalTech?")
            
            assert "GlobalTech" in response.answer

    def test_should_handle_llm_error(self) -> None:
        """Should handle LLM errors gracefully."""
        with patch("src.llm.chains.OllamaLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            
            from src.llm.chains import ReasoningChain
            
            chain = ReasoningChain()
            
            result = RetrievalResult(query="test")
            response = chain.reason(result, "test question")
            
            assert "error" in response.answer.lower()


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_reasoning_prompt_includes_context(self) -> None:
        """Should include context in reasoning prompt."""
        from src.llm.chains import REASONING_PROMPT_TEMPLATE
        
        prompt = REASONING_PROMPT_TEMPLATE.format(
            context="Test context about Singapore",
            question="What is happening?"
        )
        
        assert "Singapore" in prompt
        assert "What is happening?" in prompt

    def test_reasoning_prompt_requests_entities(self) -> None:
        """Should request entity identification."""
        from src.llm.chains import REASONING_PROMPT_TEMPLATE
        
        assert "<entities>" in REASONING_PROMPT_TEMPLATE
        assert "<reasoning>" in REASONING_PROMPT_TEMPLATE
        assert "<answer>" in REASONING_PROMPT_TEMPLATE
