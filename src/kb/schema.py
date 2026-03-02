from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConceptEntry(BaseModel):
    """Schema for a single DSA concept in the knowledge base."""

    concept_id: str = Field(description="Unique identifier, e.g. 'dsa-binary-search-tree'")
    concept_name: str = Field(description="Human-readable name, e.g. 'Binary Search Tree'")
    definition: str = Field(description="2-4 sentence definition of the concept")
    key_properties: list[str] = Field(
        description="Properties a correct answer must address"
    )
    common_misconceptions: list[str] = Field(
        default_factory=list,
        description="Documented error patterns candidates commonly exhibit",
    )
    example_correct_response: str = Field(
        description="Reference answer at expected competency level"
    )
    difficulty_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Concept difficulty tier"
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="List of related concept_ids",
    )
