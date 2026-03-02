"""Tests for DMA agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.dma import DialogueManagerAgent
from src.config import Settings
from src.kb.retriever import KBRetriever
from src.models import IntentType, TopicState


@pytest.fixture
def dma(mock_chroma_collection, mock_openai_client, mock_llm):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    settings = Settings(openai_api_key="test", anthropic_api_key="test")
    return DialogueManagerAgent(mock_llm, retriever, settings)


def test_initialize_session(dma):
    state = dma.initialize_session(topic_concept_ids=["dsa-array", "dsa-heap"])
    assert state["current_topic"] is not None
    assert state["current_topic"].concept_id == "dsa-array"
    assert state["topic_queue"] == ["dsa-heap"]
    assert state["session_complete"] is False


def test_advance_topic(dma):
    current = TopicState(concept_id="dsa-array", concept_name="Array")
    queue = ["dsa-heap", "dsa-trie"]
    result = dma.advance_topic(current, queue)
    assert result["current_topic"].concept_id == "dsa-heap"
    assert result["topic_queue"] == ["dsa-trie"]
    assert current.status == "completed"


def test_advance_topic_empty_queue(dma):
    current = TopicState(concept_id="dsa-array", concept_name="Array")
    result = dma.advance_topic(current, [])
    assert result["current_topic"] is None
    assert result["topic_queue"] == []


def test_should_end_session_max_turns(dma):
    topic = TopicState(concept_id="dsa-array", concept_name="Array")
    assert dma.should_end_session(topic, ["dsa-heap"], 30, 30) is True


def test_should_end_session_no_topics(dma):
    assert dma.should_end_session(None, [], 5, 30) is True


def test_should_end_session_continues(dma):
    topic = TopicState(concept_id="dsa-array", concept_name="Array")
    assert dma.should_end_session(topic, ["dsa-heap"], 5, 30) is False


@pytest.mark.asyncio
async def test_classify_intent(dma):
    dma.llm.ainvoke.return_value = MagicMock(content="substantive_response")
    intent = await dma.classify_intent("Arrays use contiguous memory.", "What is an array?")
    assert intent == IntentType.SUBSTANTIVE_RESPONSE


@pytest.mark.asyncio
async def test_classify_intent_clarification(dma):
    dma.llm.ainvoke.return_value = MagicMock(content="clarification_question")
    intent = await dma.classify_intent("Can you rephrase?", "Explain BST.")
    assert intent == IntentType.CLARIFICATION_QUESTION
