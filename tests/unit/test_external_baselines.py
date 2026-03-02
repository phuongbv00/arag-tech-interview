"""Tests for external baselines."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.baselines.external.cot_single_agent import CoTSingleAgentInterviewer
from src.baselines.external.single_llm import SingleLLMInterviewer
from src.baselines.external.static_question_bank import StaticQuestionBankInterviewer
from src.simulation.personas import PersonaProfile


@pytest.fixture
def mock_candidate():
    """Mock SyntheticCandidate that returns canned responses."""
    candidate = AsyncMock()
    candidate.respond = AsyncMock(
        return_value="An array stores elements in contiguous memory with O(1) access."
    )
    return candidate


@pytest.fixture
def topic_ids():
    return ["dsa-array", "dsa-linked-list"]


# --- SingleLLMInterviewer ---


@pytest.mark.asyncio
async def test_single_llm_completes_interview(mock_llm, mock_candidate, topic_ids):
    """SingleLLMInterviewer runs a full interview and returns results."""
    call_count = 0

    async def fake_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        # First 2 calls: ask questions. Third call: final assessment.
        if call_count <= 2:
            return MagicMock(content="What can you tell me about arrays?")
        return MagicMock(
            content=json.dumps(
                {
                    "done": True,
                    "assessments": [
                        {
                            "concept_id": "dsa-array",
                            "question_text": "What is an array?",
                            "overall_score": 0.8,
                            "strengths": ["Good understanding of indexing"],
                            "gaps": ["Missing memory layout"],
                            "misconceptions": [],
                        }
                    ],
                    "overall_summary": "Good interview.",
                    "strengths": ["Solid basics"],
                    "areas_for_improvement": ["Memory concepts"],
                }
            )
        )

    mock_llm.ainvoke = fake_ainvoke

    interviewer = SingleLLMInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate, max_turns=10)

    assert len(result.assessments) >= 1
    assert result.assessments[0].concept_id == "dsa-array"
    assert result.assessments[0].overall_score == 0.8
    assert result.feedback is not None
    assert len(result.conversation_history) > 0


@pytest.mark.asyncio
async def test_single_llm_forces_assessment_on_timeout(mock_llm, mock_candidate, topic_ids):
    """If the LLM never outputs final JSON, force assessment at the end."""
    call_count = 0

    async def fake_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        # Last call (forced): return final assessment
        if call_count > 4:
            return MagicMock(
                content=json.dumps(
                    {
                        "done": True,
                        "assessments": [
                            {
                                "concept_id": "dsa-array",
                                "question_text": "Q",
                                "overall_score": 0.5,
                                "strengths": [],
                                "gaps": [],
                                "misconceptions": [],
                            }
                        ],
                    }
                )
            )
        return MagicMock(content="Tell me about arrays.")

    mock_llm.ainvoke = fake_ainvoke

    interviewer = SingleLLMInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate, max_turns=3)

    assert len(result.assessments) >= 1


# --- StaticQuestionBankInterviewer ---


@pytest.mark.asyncio
async def test_static_bank_asks_one_per_topic(mock_llm, mock_candidate, topic_ids):
    """Static bank asks exactly one question per topic, no follow-ups."""
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "overall_score": 0.7,
                    "element_scores": [
                        {
                            "element": "definition",
                            "score": 0.7,
                            "justification": "Partial",
                            "grounding_source": "llm_static_bank",
                        }
                    ],
                    "misconceptions_detected": [],
                    "strengths": ["Basic understanding"],
                    "gaps": ["Missing details"],
                }
            )
        )
    )

    interviewer = StaticQuestionBankInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate, max_turns=10)

    # Exactly one question per topic
    assert len(result.questions_asked) == len(topic_ids)
    assert len(result.assessments) == len(topic_ids)
    assert result.feedback is not None

    # No follow-ups: each topic appears exactly once
    asked_topics = [t for t, _ in result.questions_asked]
    assert asked_topics == topic_ids


@pytest.mark.asyncio
async def test_static_bank_no_grounding_sources(mock_llm, mock_candidate, topic_ids):
    """Static bank assessments have no KB grounding sources."""
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "overall_score": 0.6,
                    "element_scores": [],
                    "misconceptions_detected": [],
                }
            )
        )
    )

    interviewer = StaticQuestionBankInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate)

    for assessment in result.assessments:
        assert assessment.grounding_sources == []


# --- CoTSingleAgentInterviewer ---


@pytest.mark.asyncio
async def test_cot_agent_produces_structured_output(mock_llm, mock_candidate, topic_ids):
    """CoT agent produces assessments and feedback from structured JSON."""
    call_count = 0

    async def fake_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return MagicMock(
                content=json.dumps(
                    {
                        "reasoning": {
                            "step1_gap_analysis": "First question",
                            "step2_decision": "Ask about arrays",
                            "step3_justification": "Starting topic",
                        },
                        "action": "ask_question",
                        "current_topic": "dsa-array",
                        "question": "Explain arrays and their time complexity.",
                        "assessment_of_last_response": None,
                    }
                )
            )
        return MagicMock(
            content=json.dumps(
                {
                    "reasoning": {
                        "step1_gap_analysis": "Candidate covered basics",
                        "step2_decision": "End interview",
                        "step3_justification": "All topics covered",
                    },
                    "action": "end_interview",
                    "assessment_of_last_response": {
                        "concept_id": "dsa-array",
                        "overall_score": 0.75,
                        "elements": [
                            {
                                "element": "O(1) access",
                                "score": 1.0,
                                "justification": "Correctly stated",
                            }
                        ],
                        "misconceptions": [],
                    },
                    "final_feedback": {
                        "overall_summary": "Good basic knowledge.",
                        "topic_scores": [
                            {
                                "concept_id": "dsa-array",
                                "score": 0.75,
                                "strengths": ["Index access"],
                                "gaps": ["Memory layout"],
                            }
                        ],
                        "strengths": ["Fundamentals"],
                        "areas_for_improvement": ["Advanced topics"],
                    },
                }
            )
        )

    mock_llm.ainvoke = fake_ainvoke

    interviewer = CoTSingleAgentInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate, max_turns=10)

    assert len(result.assessments) >= 1
    assert result.feedback is not None
    assert result.feedback.overall_summary == "Good basic knowledge."
    assert len(result.questions_asked) > 0

    # No KB grounding
    for assessment in result.assessments:
        assert assessment.grounding_sources == []


@pytest.mark.asyncio
async def test_cot_agent_handles_json_parse_failure(mock_llm, mock_candidate, topic_ids):
    """CoT agent gracefully handles non-JSON LLM output."""
    call_count = 0

    async def fake_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content="Sure, let me start the interview about arrays.")
        if call_count == 2:
            return MagicMock(
                content=json.dumps(
                    {
                        "action": "end_interview",
                        "assessment_of_last_response": None,
                        "final_feedback": {
                            "overall_summary": "Done.",
                            "topic_scores": [],
                            "strengths": [],
                            "areas_for_improvement": [],
                        },
                    }
                )
            )
        return MagicMock(content="{}")

    mock_llm.ainvoke = fake_ainvoke

    interviewer = CoTSingleAgentInterviewer(mock_llm)
    result = await interviewer.run_interview(topic_ids, mock_candidate, max_turns=5)

    # Should complete without crashing
    assert result.feedback is not None
