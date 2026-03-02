"""Custom evaluation metrics for the interview system."""

from __future__ import annotations

from src.kb.schema import ConceptEntry
from src.models import AssessmentRecord, ElementScore, FeedbackReport
from src.simulation.personas import PersonaProfile


def gap_coverage_rate(
    detected_gaps: list[str],
    planted_gaps: list[str],
) -> float:
    """Proportion of planted gaps that were detected and probed.

    Target >= 0.80.

    Args:
        detected_gaps: concept_ids where the system identified and probed gaps.
        planted_gaps: concept_ids where gaps were planted in the persona profile.
    """
    if not planted_gaps:
        return 1.0
    detected_set = set(detected_gaps)
    covered = sum(1 for g in planted_gaps if g in detected_set)
    return covered / len(planted_gaps)


def redundancy_rate(
    questions_asked: list[tuple[str, str]],
    concepts_addressed: dict[str, set[str]],
) -> float:
    """Proportion of questions that overlap already-addressed content.

    Target <= 0.10.

    Args:
        questions_asked: list of (concept_id, question_text) tuples in order.
        concepts_addressed: concept_id -> set of properties already addressed
            at the time the question was asked.
    """
    if not questions_asked:
        return 0.0

    redundant = 0
    for concept_id, _ in questions_asked:
        addressed = concepts_addressed.get(concept_id, set())
        if addressed:
            redundant += 1

    return redundant / len(questions_asked)


def element_precision_recall(
    assessments: list[AssessmentRecord],
    profile: PersonaProfile,
) -> tuple[float, float]:
    """Element-level precision and recall vs. ground truth persona profile.

    Precision: Of the gaps the system identified, how many are real?
    Recall: Of the real gaps, how many did the system identify?

    Args:
        assessments: All assessment records from the session.
        profile: The ground truth persona profile.
    """
    real_gaps = set(profile.partial_concepts + profile.unknown_concepts)
    real_mastered = set(profile.mastered_concepts)

    system_identified_gaps: set[str] = set()
    system_identified_mastered: set[str] = set()

    for assessment in assessments:
        if assessment.overall_score < 0.5:
            system_identified_gaps.add(assessment.concept_id)
        else:
            system_identified_mastered.add(assessment.concept_id)

    # Precision: correctly identified gaps / total identified gaps
    true_positive_gaps = system_identified_gaps & real_gaps
    precision = (
        len(true_positive_gaps) / len(system_identified_gaps)
        if system_identified_gaps
        else 1.0
    )

    # Recall: correctly identified gaps / total real gaps
    assessed_real_gaps = real_gaps & (system_identified_gaps | system_identified_mastered)
    recall = (
        len(true_positive_gaps) / len(assessed_real_gaps)
        if assessed_real_gaps
        else 1.0
    )

    return precision, recall


def hallucination_rate(
    assessments: list[AssessmentRecord],
    kb_concept_ids: set[str],
) -> float:
    """Proportion of assessment justifications citing non-existent KB entries.

    Args:
        assessments: All assessment records.
        kb_concept_ids: Set of valid concept_ids in the knowledge base.
    """
    total_sources = 0
    invalid_sources = 0

    for assessment in assessments:
        for source in assessment.grounding_sources:
            total_sources += 1
            if source not in kb_concept_ids:
                invalid_sources += 1

    if total_sources == 0:
        return 0.0
    return invalid_sources / total_sources


def evidence_citation_rate(feedback: FeedbackReport) -> float:
    """Proportion of feedback topic summaries that cite at least one KB entry.

    Args:
        feedback: The feedback report from FSA.
    """
    if not feedback.topic_summaries:
        return 0.0

    cited = sum(
        1 for ts in feedback.topic_summaries if ts.cited_sources
    )
    return cited / len(feedback.topic_summaries)
