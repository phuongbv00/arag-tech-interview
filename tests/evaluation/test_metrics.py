"""Tests for custom evaluation metrics."""

from src.evaluation.metrics import (
    element_precision_recall,
    evidence_citation_rate,
    gap_coverage_rate,
    hallucination_rate,
)
from src.models import (
    AssessmentRecord,
    ElementScore,
    FeedbackReport,
    TopicFeedback,
)
from src.simulation.personas import PersonaProfile


def test_gap_coverage_rate_all_detected():
    detected = ["dsa-trie", "dsa-red-black-tree", "dsa-b-tree"]
    planted = ["dsa-trie", "dsa-red-black-tree", "dsa-b-tree"]
    assert gap_coverage_rate(detected, planted) == 1.0


def test_gap_coverage_rate_partial():
    detected = ["dsa-trie"]
    planted = ["dsa-trie", "dsa-red-black-tree"]
    assert gap_coverage_rate(detected, planted) == 0.5


def test_gap_coverage_rate_none():
    assert gap_coverage_rate([], ["dsa-trie"]) == 0.0


def test_gap_coverage_rate_no_planted():
    assert gap_coverage_rate([], []) == 1.0


def test_hallucination_rate_no_hallucinations():
    assessments = [
        AssessmentRecord(
            concept_id="dsa-array",
            question_text="Q",
            response_text="A",
            element_scores=[],
            overall_score=0.8,
            grounding_sources=["dsa-array"],
        )
    ]
    kb_ids = {"dsa-array", "dsa-heap"}
    assert hallucination_rate(assessments, kb_ids) == 0.0


def test_hallucination_rate_with_hallucinations():
    assessments = [
        AssessmentRecord(
            concept_id="dsa-array",
            question_text="Q",
            response_text="A",
            element_scores=[],
            overall_score=0.8,
            grounding_sources=["dsa-array", "dsa-fake"],
        )
    ]
    kb_ids = {"dsa-array"}
    assert hallucination_rate(assessments, kb_ids) == 0.5


def test_evidence_citation_rate_all_cited():
    feedback = FeedbackReport(
        overall_summary="Good",
        topic_summaries=[
            TopicFeedback(
                concept_name="Array",
                score=0.8,
                strengths=["Good"],
                gaps=[],
                recommendations=[],
                cited_sources=["dsa-array"],
            ),
        ],
        strengths=["Good"],
        areas_for_improvement=[],
    )
    assert evidence_citation_rate(feedback) == 1.0


def test_evidence_citation_rate_none_cited():
    feedback = FeedbackReport(
        overall_summary="Good",
        topic_summaries=[
            TopicFeedback(
                concept_name="Array",
                score=0.8,
                strengths=["Good"],
                gaps=[],
                recommendations=[],
                cited_sources=[],
            ),
        ],
        strengths=["Good"],
        areas_for_improvement=[],
    )
    assert evidence_citation_rate(feedback) == 0.0


def test_element_precision_recall():
    profile = PersonaProfile(
        level="L1",
        mastered_concepts=["dsa-array"],
        partial_concepts=["dsa-heap"],
        unknown_concepts=["dsa-trie"],
    )
    assessments = [
        AssessmentRecord(
            concept_id="dsa-array",
            question_text="Q",
            response_text="A",
            element_scores=[],
            overall_score=0.9,  # correctly identified as mastered
            grounding_sources=["dsa-array"],
        ),
        AssessmentRecord(
            concept_id="dsa-heap",
            question_text="Q",
            response_text="A",
            element_scores=[],
            overall_score=0.3,  # correctly identified as gap
            grounding_sources=["dsa-heap"],
        ),
    ]
    precision, recall = element_precision_recall(assessments, profile)
    assert precision == 1.0  # identified gap (heap) is a real gap
    assert recall == 1.0  # all assessed real gaps were detected
