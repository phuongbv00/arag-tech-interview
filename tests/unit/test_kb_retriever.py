"""Tests for KB retriever."""

from unittest.mock import MagicMock

import chromadb

from src.kb.retriever import KBRetriever
from src.kb.schema import ConceptEntry


def test_get_by_concept_id(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    result = retriever.get_by_concept_id("dsa-array")
    assert result is not None
    assert result.concept_id == "dsa-array"
    assert result.concept_name == "Array"


def test_get_by_concept_id_not_found(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    result = retriever.get_by_concept_id("dsa-nonexistent")
    assert result is None


def test_get_by_difficulty(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    beginners = retriever.get_by_difficulty("beginner")
    assert len(beginners) == 2  # array and linked-list
    assert all(c.difficulty_level == "beginner" for c in beginners)


def test_get_by_difficulty_with_exclude(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    beginners = retriever.get_by_difficulty("beginner", exclude_ids=["dsa-array"])
    assert len(beginners) == 1
    assert beginners[0].concept_id == "dsa-linked-list"


def test_get_all_concepts(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    all_concepts = retriever.get_all_concepts()
    assert len(all_concepts) == 5


def test_get_related_concepts(mock_chroma_collection, mock_openai_client):
    retriever = KBRetriever(mock_chroma_collection, mock_openai_client)
    related = retriever.get_related_concepts("dsa-array")
    related_ids = [c.concept_id for c in related]
    assert "dsa-linked-list" in related_ids
