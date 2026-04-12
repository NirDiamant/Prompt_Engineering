"""Integration tests for MiniMax provider.

These tests make real API calls to the MiniMax API and require
the MINIMAX_API_KEY environment variable to be set.

Run with:
    pytest tests/test_integration_minimax.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_provider import get_llm

# Skip all tests if MINIMAX_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


class TestMiniMaxIntegration:
    """Integration tests that call the MiniMax API."""

    def test_basic_completion(self):
        """MiniMax should return a non-empty response for a simple prompt."""
        llm = get_llm(provider="minimax", model="MiniMax-M2.5")
        result = llm.invoke("Say hello in exactly three words.").content
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_sentiment_classification(self):
        """MiniMax should correctly classify obvious sentiment."""
        llm = get_llm(provider="minimax", model="MiniMax-M2.5")
        result = llm.invoke(
            "Classify the sentiment as Positive, Negative, or Neutral. "
            "Respond with only the label.\n\n"
            "Text: I absolutely love this product!\n\n"
            "Sentiment:"
        ).content.strip().lower()
        assert "positive" in result

    def test_chain_with_prompt_template(self):
        """MiniMax should work with LangChain PromptTemplate chains."""
        from langchain_core.prompts import PromptTemplate

        llm = get_llm(provider="minimax")
        prompt = PromptTemplate.from_template(
            "What is the capital of {country}? Reply with just the city name."
        )
        chain = prompt | llm
        result = chain.invoke({"country": "France"}).content.strip()
        assert "Paris" in result
