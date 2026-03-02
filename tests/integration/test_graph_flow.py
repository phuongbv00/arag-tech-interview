"""Integration tests for the LangGraph interview flow."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.dma import DialogueManagerAgent
from src.config import Settings
from src.kb.retriever import KBRetriever
from src.models import DMAAction, IntentType


@pytest.fixture
def settings():
    return Settings(
        openai_api_key="test",
        anthropic_api_key="test",
        max_topics=2,
        max_depth_per_topic=2,
        max_total_turns=10,
    )


@pytest.fixture
def retriever(mock_chroma_collection, mock_openai_client):
    return KBRetriever(mock_chroma_collection, mock_openai_client)


def test_session_initialization(retriever, settings, mock_llm):
    """Test that session initializes with correct state."""
    dma = DialogueManagerAgent(mock_llm, retriever, settings)
    state = dma.initialize_session(
        topic_concept_ids=["dsa-array", "dsa-binary-search-tree"]
    )

    assert state["current_topic"].concept_id == "dsa-array"
    assert state["topic_queue"] == ["dsa-binary-search-tree"]
    assert state["session_complete"] is False
    assert state["session_turn_count"] == 0


def test_full_topic_cycle(retriever, settings, mock_llm):
    """Test advancing through topics."""
    dma = DialogueManagerAgent(mock_llm, retriever, settings)
    state = dma.initialize_session(
        topic_concept_ids=["dsa-array", "dsa-heap"]
    )

    # Advance from array to heap
    update = dma.advance_topic(state["current_topic"], state["topic_queue"])
    assert update["current_topic"].concept_id == "dsa-heap"
    assert update["topic_queue"] == []

    # Advance from heap to nothing
    update2 = dma.advance_topic(update["current_topic"], update["topic_queue"])
    assert update2["current_topic"] is None


@pytest.mark.asyncio
async def test_intent_classification_flow(retriever, settings):
    """Test that intent classification works through DMA."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="substantive_response")

    dma = DialogueManagerAgent(mock_llm, retriever, settings)
    intent = await dma.classify_intent(
        "An array stores elements in contiguous memory.",
        "What is an array?",
    )
    assert intent == IntentType.SUBSTANTIVE_RESPONSE


@pytest.mark.asyncio
async def test_action_decision_flow(retriever, settings):
    """Test that action decision works through DMA."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="delegate_kgda")

    dma = DialogueManagerAgent(mock_llm, retriever, settings)
    action = await dma.decide_action(
        topic_name="Array",
        depth=0,
        max_depth=2,
        topics_remaining=1,
        turns_used=1,
        max_turns=10,
        intent=IntentType.SUBSTANTIVE_RESPONSE,
        gap_summary="No gap report yet",
    )
    assert action == DMAAction.DELEGATE_KGDA
