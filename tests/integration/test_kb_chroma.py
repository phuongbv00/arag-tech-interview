"""Integration tests for ChromaDB round-trip."""

import chromadb

from src.kb.retriever import KBRetriever
from src.kb.schema import ConceptEntry


def test_chroma_roundtrip(mock_kb, mock_openai_client):
    """Seed a small KB, verify retrieval round-trip."""
    client = chromadb.Client()
    collection = client.create_collection(name="test_roundtrip")

    ids = [c.concept_id for c in mock_kb]
    documents = [f"{c.concept_name}\n{c.definition}" for c in mock_kb]
    metadatas = [c.model_dump() for c in mock_kb]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    retriever = KBRetriever(collection, mock_openai_client)

    # Test direct lookup
    result = retriever.get_by_concept_id("dsa-array")
    assert result is not None
    assert result.concept_name == "Array"
    assert "O(1) access by index" in result.key_properties

    # Test difficulty filter
    intermediates = retriever.get_by_difficulty("intermediate")
    assert len(intermediates) == 2
    ids_found = {c.concept_id for c in intermediates}
    assert "dsa-binary-search-tree" in ids_found
    assert "dsa-heap" in ids_found

    # Test related concepts
    related = retriever.get_related_concepts("dsa-array")
    related_ids = {c.concept_id for c in related}
    assert "dsa-linked-list" in related_ids

    # Test get all
    all_concepts = retriever.get_all_concepts()
    assert len(all_concepts) == 5
