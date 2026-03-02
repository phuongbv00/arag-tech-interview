"""Tests for LangGraph state schema."""

from src.models import (
    AssessmentRecord,
    DMAAction,
    ElementScore,
    FeedbackReport,
    GapItem,
    GapReport,
    IntentType,
    TopicFeedback,
    TopicState,
)


def test_intent_type_values():
    assert IntentType.SUBSTANTIVE_RESPONSE.value == "substantive_response"
    assert IntentType.CLARIFICATION_QUESTION.value == "clarification_question"
    assert IntentType.NON_ANSWER.value == "non_answer"
    assert IntentType.REQUEST_HINT.value == "request_hint"


def test_dma_action_values():
    assert DMAAction.DELEGATE_KGDA.value == "delegate_kgda"
    assert DMAAction.DELEGATE_QGA.value == "delegate_qga"
    assert DMAAction.RESPOND_DIRECTLY.value == "respond_directly"
    assert DMAAction.MOVE_ON.value == "move_on"
    assert DMAAction.TRIGGER_FSA.value == "trigger_fsa"


def test_gap_report_creation():
    report = GapReport(
        concept_id="dsa-array",
        concept_name="Array",
        items=[
            GapItem(
                property_text="O(1) access",
                status="addressed_correctly",
                evidence="mentioned O(1)",
            ),
            GapItem(
                property_text="contiguous memory",
                status="not_addressed",
            ),
        ],
        priority_gap="contiguous memory",
        retrieval_sources=["dsa-array"],
    )
    assert len(report.items) == 2
    assert report.priority_gap == "contiguous memory"


def test_topic_state_defaults():
    topic = TopicState(concept_id="dsa-array", concept_name="Array")
    assert topic.depth == 0
    assert topic.questions_asked == []
    assert topic.status == "active"


def test_assessment_record_score_bounds():
    record = AssessmentRecord(
        concept_id="dsa-array",
        question_text="Q",
        response_text="A",
        element_scores=[
            ElementScore(
                element="elem",
                score=0.5,
                justification="partial",
                grounding_source="dsa-array",
            )
        ],
        overall_score=0.5,
        grounding_sources=["dsa-array"],
    )
    assert 0.0 <= record.overall_score <= 1.0
    assert 0.0 <= record.element_scores[0].score <= 1.0
