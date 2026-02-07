"""
LLM Chains and Prompts for Atlas-GRAG.

Provides prompt templates and chain definitions for the reasoning
layer that synthesizes answers from hybrid retrieval context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_ollama import OllamaLLM

from src.config import get_config
from src.retriever.hybrid import RetrievalResult

logger = logging.getLogger(__name__)


# Chain of Thought prompt for supply chain reasoning
REASONING_PROMPT_TEMPLATE = """You are Atlas-GRAG, an expert supply chain risk analyst AI.
Your task is to answer questions about global supply chains by reasoning through
the knowledge provided from both documents AND a knowledge graph.

IMPORTANT INSTRUCTIONS:
1. Use the Knowledge Graph relationships to trace multi-hop connections
2. Cite specific relationships when explaining your reasoning
3. If the question requires connecting multiple facts, show your chain of thought
4. If information is missing, say so clearly

## Retrieved Context

{context}

## User Question
{question}

## Your Analysis

First, identify the relevant entities and their connections:
<entities>
List the key entities mentioned in the context that relate to the question.
</entities>

<reasoning>
Walk through the chain of relationships step by step.
For example: "Entity A is connected to Entity B via [RELATIONSHIP], and Entity B is connected to Entity C via [RELATIONSHIP]."
</reasoning>

<answer>
Provide your final answer based on the traced relationships.
</answer>

Begin your analysis:"""


SIMPLE_ANSWER_PROMPT = """Based on the following context, answer the user's question.
If you cannot find the answer in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class ReasoningResponse:
    """
    Response from the reasoning chain.
    
    Attributes:
        answer: The final answer text
        entities: Entities identified in reasoning
        reasoning: Chain of thought reasoning
        raw_response: Full LLM response
    """
    answer: str
    entities: str = ""
    reasoning: str = ""
    raw_response: str = ""
    
    @classmethod
    def parse_from_response(cls, response: str) -> "ReasoningResponse":
        """
        Parse structured response from LLM.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed ReasoningResponse
        """
        entities = ""
        reasoning = ""
        answer = response
        
        # Try to extract structured sections
        try:
            import re
            
            entities_match = re.search(r"<entities>(.*?)</entities>", response, re.DOTALL)
            if entities_match:
                entities = entities_match.group(1).strip()
            
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            elif not answer_match:
                # If no structured answer, use everything after </reasoning>
                after_reasoning = re.split(r"</reasoning>", response)
                if len(after_reasoning) > 1:
                    answer = after_reasoning[1].strip()
                    
        except Exception:
            pass
        
        return cls(
            answer=answer,
            entities=entities,
            reasoning=reasoning,
            raw_response=response
        )


class ReasoningChain:
    """
    LLM chain for reasoning over hybrid retrieval context.
    
    Implements 'Chain of Thought' prompting to force the LLM
    to cite specific graph relationships in its reasoning.
    
    Example:
        chain = ReasoningChain()
        result = retriever.retrieve("How does strike affect GlobalTech?")
        response = chain.reason(result, "How does strike affect GlobalTech?")
        print(response.answer)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> None:
        """
        Initialize the reasoning chain.
        
        Args:
            model: Ollama model name
            temperature: LLM temperature (low for factual responses)
        """
        config = get_config().ollama
        
        self._model = model or config.model
        self._llm = OllamaLLM(
            model=self._model,
            base_url=config.base_url,
            temperature=temperature
        )
    
    def reason(
        self,
        retrieval_result: RetrievalResult,
        question: str,
        use_chain_of_thought: bool = True
    ) -> ReasoningResponse:
        """
        Generate a reasoned answer from retrieval context.
        
        Args:
            retrieval_result: Context from hybrid retriever
            question: User's question
            use_chain_of_thought: Whether to use structured CoT prompt
            
        Returns:
            ReasoningResponse with answer and reasoning trace
        """
        context = retrieval_result.get_combined_context()
        
        if use_chain_of_thought:
            prompt = REASONING_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            prompt = SIMPLE_ANSWER_PROMPT.format(
                context=context,
                question=question
            )
        
        try:
            response = self._llm.invoke(prompt)
            
            if use_chain_of_thought:
                return ReasoningResponse.parse_from_response(response)
            else:
                return ReasoningResponse(answer=response, raw_response=response)
                
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasoningResponse(
                answer=f"I encountered an error while reasoning: {str(e)}",
                raw_response=str(e)
            )
    
    def simple_answer(
        self,
        context: str,
        question: str
    ) -> str:
        """
        Generate a simple answer without structured reasoning.
        
        Args:
            context: Context string
            question: User's question
            
        Returns:
            Answer string
        """
        prompt = SIMPLE_ANSWER_PROMPT.format(
            context=context,
            question=question
        )
        
        try:
            return self._llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Simple answer failed: {e}")
            return f"Error: {str(e)}"
