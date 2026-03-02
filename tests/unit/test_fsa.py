"""Tests for FSA agent."""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.fsa import FeedbackSynthesisAgent
from src.kb.retriever import KBRetriever
from src.models import AssessmentRecord, ElementScore, TopicState


@pytest.fixture
def fsa(mock_chroma_collection, mock_openai_client, mock_llm):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    return FeedbackSynthesisAgent(mock_llm, retriever)


@pytest.mark.asyncio
async def test_synthesize_returns_feedback(fsa):
    fsa.llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "overall_summary": "The candidate showed basic knowledge.",
                "topic_summaries": [
                    {
                        "concept_name": "Array",
                        "score": 0.7,
                        "strengths": ["Good understanding of indexing"],
                        "gaps": ["Missing memory layout details"],
                        "recommendations": ["Study contiguous memory allocation"],
                        "cited_sources": ["dsa-array"],
                    }
                ],
                "strengths": ["Solid fundamentals"],
                "areas_for_improvement": ["Memory management concepts"],
                "kb_references": ["dsa-array"],
            }
        )
    )

    assessments = [
        AssessmentRecord(
            concept_id="dsa-array",
            question_text="What is an array?",
            response_text="Arrays give O(1) access.",
            element_scores=[
                ElementScore(
                    element="O(1) access",
                    score=1.0,
                    justification="Correct",
                    grounding_source="dsa-array",
                )
            ],
            overall_score=0.7,
            grounding_sources=["dsa-array"],
        )
    ]
    topics = [
        TopicState(concept_id="dsa-array", concept_name="Array", status="completed")
    ]

    report = await fsa.synthesize(assessments, topics)
    assert report.overall_summary
    assert len(report.topic_summaries) == 1
    assert report.topic_summaries[0].concept_name == "Array"
    assert report.kb_references == ["dsa-array"]
