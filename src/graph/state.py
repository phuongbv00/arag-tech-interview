from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages

from src.models import (
    AssessmentRecord,
    DMAAction,
    FeedbackReport,
    GapReport,
    IntentType,
    TopicState,
)


class InterviewState(TypedDict):
    """Central LangGraph state for the interview session.

    All agents read from and write to this shared state.
    """

    # Conversation history (append-only via add_messages reducer)
    messages: Annotated[list, add_messages]

    # DMA deterministic state
    current_topic: TopicState | None
    topic_queue: list[str]  # concept_ids remaining to cover
    topics_covered: list[TopicState]  # completed topics
    session_turn_count: int
    max_turns: int

    # DMA LLM outputs
    classified_intent: IntentType | None
    dma_action: DMAAction | None
    dma_direct_response: str | None

    # Agent outputs (written by respective agents, read by DMA)
    current_gap_report: GapReport | None
    current_assessment: AssessmentRecord | None
    pending_question: str | None  # from QGA
    expected_elements: list[str]  # from QGA, forwarded to RAA

    # Session-level aggregation
    all_assessments: list[AssessmentRecord]
    feedback_report: FeedbackReport | None
    session_complete: bool
