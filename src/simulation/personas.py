"""L1/L3 persona profile builders for synthetic candidate simulation."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from src.kb.schema import ConceptEntry


class PersonaProfile(BaseModel):
    """Defines what a synthetic candidate knows and does not know."""

    level: str  # "L1" or "L3"
    mastered_concepts: list[str] = Field(
        default_factory=list, description="concept_ids the candidate fully knows"
    )
    partial_concepts: list[str] = Field(
        default_factory=list, description="concept_ids with incomplete knowledge"
    )
    unknown_concepts: list[str] = Field(
        default_factory=list, description="concept_ids the candidate cannot answer"
    )
    misconception_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="concept_id -> list of misconceptions to exhibit",
    )


def build_l1_persona(concepts: list[ConceptEntry]) -> PersonaProfile:
    """Build an L1 (Beginner) persona profile.

    - Masters ~60% of beginner concepts
    - Partial on remaining beginner concepts
    - Unknown on all intermediate and advanced concepts
    - Exhibits common misconceptions on partial/unknown concepts
    """
    by_difficulty: dict[str, list[ConceptEntry]] = {
        "beginner": [],
        "intermediate": [],
        "advanced": [],
    }
    for c in concepts:
        by_difficulty[c.difficulty_level].append(c)

    beginner = by_difficulty["beginner"]
    random.shuffle(beginner)
    mastery_count = max(1, int(len(beginner) * 0.6))

    mastered = [c.concept_id for c in beginner[:mastery_count]]
    partial = [c.concept_id for c in beginner[mastery_count:]]
    unknown = [
        c.concept_id
        for c in by_difficulty["intermediate"] + by_difficulty["advanced"]
    ]

    # Build misconception map for partial and unknown concepts
    misconception_map: dict[str, list[str]] = {}
    for c in concepts:
        if c.concept_id in partial or c.concept_id in unknown:
            if c.common_misconceptions:
                misconception_map[c.concept_id] = c.common_misconceptions

    return PersonaProfile(
        level="L1",
        mastered_concepts=mastered,
        partial_concepts=partial,
        unknown_concepts=unknown,
        misconception_map=misconception_map,
    )


def build_l3_persona(
    concepts: list[ConceptEntry], planted_gaps: int = 3
) -> PersonaProfile:
    """Build an L3 (Advanced) persona profile.

    - Full mastery of all beginner + intermediate concepts
    - Masters most advanced concepts, deliberate gaps in `planted_gaps` advanced concepts
    - May exhibit subtle misconceptions on gap concepts
    """
    by_difficulty: dict[str, list[ConceptEntry]] = {
        "beginner": [],
        "intermediate": [],
        "advanced": [],
    }
    for c in concepts:
        by_difficulty[c.difficulty_level].append(c)

    mastered = [
        c.concept_id
        for c in by_difficulty["beginner"] + by_difficulty["intermediate"]
    ]

    advanced = by_difficulty["advanced"]
    random.shuffle(advanced)
    gap_count = min(planted_gaps, len(advanced))
    gap_concepts = advanced[:gap_count]
    mastered_advanced = advanced[gap_count:]

    mastered.extend(c.concept_id for c in mastered_advanced)
    partial = [c.concept_id for c in gap_concepts]

    # Build misconception map for gap concepts
    misconception_map: dict[str, list[str]] = {}
    for c in gap_concepts:
        if c.common_misconceptions:
            misconception_map[c.concept_id] = c.common_misconceptions[:1]

    return PersonaProfile(
        level="L3",
        mastered_concepts=mastered,
        partial_concepts=partial,
        unknown_concepts=[],
        misconception_map=misconception_map,
    )
