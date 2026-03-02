from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    SUBSTANTIVE_RESPONSE = "substantive_response"
    CLARIFICATION_QUESTION = "clarification_question"
    NON_ANSWER = "non_answer"
    REQUEST_HINT = "request_hint"


class DMAAction(str, Enum):
    DELEGATE_KGDA = "delegate_kgda"
    DELEGATE_QGA = "delegate_qga"
    RESPOND_DIRECTLY = "respond_directly"
    MOVE_ON = "move_on"
    TRIGGER_FSA = "trigger_fsa"


class GapItem(BaseModel):
    """A single property-level gap assessment."""

    property_text: str
    status: Literal["addressed_correctly", "incomplete", "incorrect", "not_addressed"]
    evidence: str = Field(default="", description="Quote from candidate response")


class GapReport(BaseModel):
    """Output of KGDA: gap analysis for a single concept."""

    concept_id: str
    concept_name: str
    items: list[GapItem]
    priority_gap: str | None = Field(
        default=None, description="Most important gap to probe next"
    )
    retrieval_sources: list[str] = Field(
        default_factory=list, description="concept_ids used for grounding"
    )


class ElementScore(BaseModel):
    """Score for a single expected answer element."""

    element: str
    score: float = Field(ge=0.0, le=1.0)
    justification: str
    grounding_source: str = Field(description="concept_id used for scoring")


class AssessmentRecord(BaseModel):
    """Output of RAA: full assessment of a candidate response."""

    concept_id: str
    question_text: str
    response_text: str
    element_scores: list[ElementScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    grounding_sources: list[str] = Field(description="concept_ids cited")
    misconceptions_detected: list[str] = Field(default_factory=list)


class TopicState(BaseModel):
    """Tracks the state of a single topic during the interview."""

    concept_id: str
    concept_name: str
    depth: int = 0
    questions_asked: list[str] = Field(default_factory=list)
    gap_reports: list[GapReport] = Field(default_factory=list)
    assessments: list[AssessmentRecord] = Field(default_factory=list)
    status: Literal["active", "completed", "skipped"] = "active"


class TopicFeedback(BaseModel):
    """Per-topic feedback in the final report."""

    concept_name: str
    score: float
    strengths: list[str]
    gaps: list[str]
    recommendations: list[str]
    cited_sources: list[str]


class FeedbackReport(BaseModel):
    """Output of FSA: end-of-session feedback report."""

    overall_summary: str
    topic_summaries: list[TopicFeedback]
    strengths: list[str]
    areas_for_improvement: list[str]
    kb_references: list[str] = Field(
        default_factory=list, description="concept_ids cited in feedback"
    )
