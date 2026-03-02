"""Tests for QGA agent."""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.qga import QuestionGenerationAgent
from src.kb.retriever import KBRetriever
from src.models import GapItem, GapReport


@pytest.fixture
def qga(mock_chroma_collection, mock_openai_client, mock_llm):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    return QuestionGenerationAgent(mock_llm, retriever)


@pytest.mark.asyncio
async def test_generate_opening_question(qga):
    qga.llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "question": "Can you explain what an array is and its key properties?",
                "expected_elements": ["O(1) index access", "contiguous memory"],
            }
        )
    )

    question, elements = await qga.generate_opening_question("dsa-array")
    assert "array" in question.lower()
    assert len(elements) == 2


@pytest.mark.asyncio
async def test_generate_followup_question(qga):
    qga.llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "question": "Can you explain memory allocation in arrays?",
                "expected_elements": ["contiguous memory"],
            }
        )
    )

    gap_report = GapReport(
        concept_id="dsa-array",
        concept_name="Array",
        items=[
            GapItem(
                property_text="Contiguous memory",
                status="not_addressed",
                evidence="",
            )
        ],
        priority_gap="Contiguous memory",
    )

    question, elements = await qga.generate_followup_question(gap_report, [])
    assert question
    assert len(elements) >= 1
