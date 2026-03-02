"""Adapter to format interview session data for RAGAS evaluate()."""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

from src.models import AssessmentRecord


@dataclass
class SessionData:
    """Collected data from a single interview session."""

    assessments: list[AssessmentRecord] = field(default_factory=list)
    retrieved_contexts: dict[str, list[str]] = field(default_factory=dict)
    ground_truths: dict[str, str] = field(default_factory=dict)


def format_for_ragas(session_data: SessionData) -> Dataset:
    """Convert interview session data into RAGAS-compatible dataset.

    Maps:
    - question: interviewer question
    - answer: assessment justification (for faithfulness) or candidate response
    - contexts: retrieved KB content used by agent
    - ground_truth: expected answer from KB
    """
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for assessment in session_data.assessments:
        questions.append(assessment.question_text)
        answers.append(assessment.response_text)

        # Contexts are the retrieved KB content for this concept
        concept_contexts = session_data.retrieved_contexts.get(
            assessment.concept_id, []
        )
        contexts.append(concept_contexts)

        # Ground truth is the expected correct response
        gt = session_data.ground_truths.get(assessment.concept_id, "")
        ground_truths.append(gt)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def compute_ragas_metrics(dataset: Dataset) -> dict[str, float]:
    """Run RAGAS evaluate() and return metric scores.

    Metrics:
    - faithfulness: How faithful are assessment justifications to retrieved context?
    - answer_relevancy: How relevant is the feedback to the actual performance?
    - context_relevancy: How relevant is the retrieved context to the question?
    """
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_relevancy],
    )
    return {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_relevancy": results["context_relevancy"],
    }
