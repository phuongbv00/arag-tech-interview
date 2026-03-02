"""Tests for KGDA agent."""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.kgda import KnowledgeGapDetectionAgent
from src.kb.retriever import KBRetriever


@pytest.fixture
def kgda(mock_chroma_collection, mock_openai_client, mock_llm):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    return KnowledgeGapDetectionAgent(mock_llm, retriever)


@pytest.mark.asyncio
async def test_analyze_returns_gap_report(kgda):
    kgda.llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "items": [
                    {
                        "property_text": "O(1) access by index",
                        "status": "addressed_correctly",
                        "evidence": "mentioned O(1) access",
                    },
                    {
                        "property_text": "Contiguous memory allocation",
                        "status": "not_addressed",
                        "evidence": "",
                    },
                ],
                "priority_gap": "Contiguous memory allocation",
            }
        )
    )

    report = await kgda.analyze(
        "Arrays give O(1) access by index.", "dsa-array"
    )

    assert report.concept_id == "dsa-array"
    assert len(report.items) == 2
    assert report.items[0].status == "addressed_correctly"
    assert report.items[1].status == "not_addressed"
    assert report.priority_gap == "Contiguous memory allocation"


@pytest.mark.asyncio
async def test_analyze_nonexistent_concept(kgda):
    report = await kgda.analyze("Some response", "dsa-nonexistent")
    assert report.items == []
    assert report.priority_gap is None
