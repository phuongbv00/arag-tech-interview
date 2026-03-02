"""Tests for KB schema validation."""

import pytest
from pydantic import ValidationError

from src.kb.schema import ConceptEntry


def test_valid_concept_entry():
    entry = ConceptEntry(
        concept_id="dsa-array",
        concept_name="Array",
        definition="A contiguous memory block.",
        key_properties=["O(1) index access"],
        common_misconceptions=["Always fixed size"],
        example_correct_response="An array stores elements contiguously.",
        difficulty_level="beginner",
        related_concepts=["dsa-linked-list"],
    )
    assert entry.concept_id == "dsa-array"
    assert entry.difficulty_level == "beginner"


def test_invalid_difficulty_level():
    with pytest.raises(ValidationError):
        ConceptEntry(
            concept_id="dsa-test",
            concept_name="Test",
            definition="A test.",
            key_properties=["prop"],
            example_correct_response="Answer.",
            difficulty_level="expert",
        )


def test_defaults():
    entry = ConceptEntry(
        concept_id="dsa-test",
        concept_name="Test",
        definition="A test concept.",
        key_properties=["prop1"],
        example_correct_response="Answer.",
        difficulty_level="beginner",
    )
    assert entry.common_misconceptions == []
    assert entry.related_concepts == []
