"""Shared fixtures for tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import chromadb
import pytest

from src.kb.schema import ConceptEntry


@pytest.fixture
def mock_kb() -> list[ConceptEntry]:
    """5 sample DSA concepts for testing."""
    return [
        ConceptEntry(
            concept_id="dsa-array",
            concept_name="Array",
            definition="A contiguous block of memory storing elements of the same type.",
            key_properties=[
                "O(1) access by index",
                "Contiguous memory allocation",
                "Fixed size (static) or resizable (dynamic)",
                "O(n) insertion/deletion in the middle",
            ],
            common_misconceptions=[
                "Arrays always have fixed size",
                "Array access is O(n)",
            ],
            example_correct_response=(
                "An array is a data structure that stores elements in contiguous "
                "memory locations, allowing O(1) access by index. Arrays can be "
                "static (fixed size) or dynamic (resizable). Insertion and deletion "
                "in the middle require shifting elements, making them O(n)."
            ),
            difficulty_level="beginner",
            related_concepts=["dsa-linked-list", "dsa-hash-table"],
        ),
        ConceptEntry(
            concept_id="dsa-linked-list",
            concept_name="Linked List",
            definition="A linear data structure where elements are stored in nodes connected by pointers.",
            key_properties=[
                "Dynamic size",
                "O(1) insertion/deletion at head",
                "O(n) access by index",
                "Non-contiguous memory",
            ],
            common_misconceptions=[
                "Linked lists have O(1) access by index",
                "Linked lists use contiguous memory",
            ],
            example_correct_response=(
                "A linked list stores elements in nodes, each containing data and a "
                "pointer to the next node. It supports O(1) insertion at the head "
                "but requires O(n) traversal for index access."
            ),
            difficulty_level="beginner",
            related_concepts=["dsa-array", "dsa-stack"],
        ),
        ConceptEntry(
            concept_id="dsa-binary-search-tree",
            concept_name="Binary Search Tree",
            definition="A binary tree where each node's left subtree contains smaller values and right subtree contains larger values.",
            key_properties=[
                "Left child < parent < right child (BST property)",
                "O(log n) average search/insert/delete",
                "O(n) worst case (degenerate tree)",
                "In-order traversal yields sorted sequence",
            ],
            common_misconceptions=[
                "BST operations are always O(log n)",
                "BST is always balanced",
            ],
            example_correct_response=(
                "A BST maintains the property that all left descendants are smaller "
                "and all right descendants are larger. This enables O(log n) average "
                "case operations, but O(n) worst case in degenerate trees."
            ),
            difficulty_level="intermediate",
            related_concepts=["dsa-avl-tree", "dsa-binary-search"],
        ),
        ConceptEntry(
            concept_id="dsa-heap",
            concept_name="Heap",
            definition="A complete binary tree satisfying the heap property.",
            key_properties=[
                "Complete binary tree structure",
                "Min-heap: parent <= children; Max-heap: parent >= children",
                "O(log n) insertion and extraction",
                "O(1) access to min/max element",
            ],
            common_misconceptions=[
                "Heaps are sorted arrays",
                "Heap operations are O(n)",
            ],
            example_correct_response=(
                "A heap is a complete binary tree where each parent satisfies the "
                "heap property. In a min-heap, every parent is smaller than its "
                "children. Insertion and extraction are O(log n), while accessing "
                "the top element is O(1)."
            ),
            difficulty_level="intermediate",
            related_concepts=["dsa-binary-tree", "dsa-array"],
        ),
        ConceptEntry(
            concept_id="dsa-trie",
            concept_name="Trie",
            definition="A tree-like data structure for efficient retrieval of keys, typically strings.",
            key_properties=[
                "O(m) search where m is key length",
                "Prefix-based organization",
                "Space-efficient for shared prefixes",
                "Supports prefix queries efficiently",
            ],
            common_misconceptions=[
                "Tries are always more space-efficient than hash tables",
                "Trie search is O(n) where n is number of keys",
            ],
            example_correct_response=(
                "A trie stores keys character by character in a tree structure. "
                "Search time is O(m) where m is the key length. Tries are "
                "particularly useful for prefix-based queries and autocomplete."
            ),
            difficulty_level="advanced",
            related_concepts=["dsa-hash-table", "dsa-string-basics"],
        ),
    ]


@pytest.fixture
def mock_chroma_collection(mock_kb: list[ConceptEntry]) -> chromadb.Collection:
    """Ephemeral ChromaDB collection seeded with mock_kb."""
    client = chromadb.Client()
    collection = client.create_collection(name="test_concepts")

    ids = [c.concept_id for c in mock_kb]
    documents = [
        f"{c.concept_name}\n{c.definition}\n{'; '.join(c.key_properties)}"
        for c in mock_kb
    ]
    metadatas = [c.model_dump() for c in mock_kb]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for testing."""
    client = MagicMock()

    embedding_response = MagicMock()
    embedding_item = MagicMock()
    embedding_item.embedding = [0.1] * 1536
    embedding_response.data = [embedding_item]
    client.embeddings.create.return_value = embedding_response

    return client


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLM for testing agent LLM calls."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = "{}"
    llm.ainvoke.return_value = response
    return llm
