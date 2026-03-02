"""Tests for RAA agent."""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.raa import ResponseAssessmentAgent
from src.kb.retriever import KBRetriever


@pytest.fixture
def raa(mock_chroma_collection, mock_openai_client, mock_llm):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    return ResponseAssessmentAgent(mock_llm, retriever)


@pytest.mark.asyncio
async def test_assess_returns_record(raa):
    raa.llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "element_scores": [
                    {
                        "element": "O(1) index access",
                        "score": 1.0,
                        "justification": "Correctly stated O(1) access",
                        "grounding_source": "dsa-array",
                    },
                    {
                        "element": "contiguous memory",
                        "score": 0.0,
                        "justification": "Not mentioned",
                        "grounding_source": "dsa-array",
                    },
                ],
                "overall_score": 0.5,
                "misconceptions_detected": [],
            }
        )
    )

    record = await raa.assess(
        candidate_response="Arrays have O(1) access by index.",
        expected_elements=["O(1) index access", "contiguous memory"],
        concept_id="dsa-array",
        question_text="What is an array?",
    )

    assert record.concept_id == "dsa-array"
    assert len(record.element_scores) == 2
    assert record.overall_score == 0.5
    assert record.grounding_sources == ["dsa-array"]


@pytest.mark.asyncio
async def test_assess_nonexistent_concept(raa):
    record = await raa.assess(
        candidate_response="Some response",
        expected_elements=["elem"],
        concept_id="dsa-nonexistent",
        question_text="Question?",
    )
    assert record.overall_score == 0.0
    assert record.element_scores == []
