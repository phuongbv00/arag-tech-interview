"""Non-RAG baseline agent variants.

Same architecture but template-based questions, LLM-only assessment.
Matches Pathak & Pandey (2025) design pattern.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import (
    AssessmentRecord,
    ElementScore,
    FeedbackReport,
    GapItem,
    GapReport,
    TopicFeedback,
    TopicState,
)


class NoRAGKGDA:
    """LLM-only gap detection without KB retrieval."""

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever  # kept for interface compatibility

    async def analyze(self, candidate_response: str, concept_id: str) -> GapReport:
        messages = [
            SystemMessage(
                content=(
                    "You are evaluating a candidate's technical interview response. "
                    "Identify what key properties they addressed correctly, "
                    "what was incomplete, incorrect, or not addressed. "
                    "Respond in JSON with 'items' (list of {property_text, status, evidence}) "
                    "and 'priority_gap' (string or null)."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {concept_id}\n\n"
                    f"Candidate response:\n{candidate_response}\n\n"
                    "Analyze the response."
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())
        items = [GapItem.model_validate(item) for item in parsed["items"]]
        return GapReport(
            concept_id=concept_id,
            concept_name=concept_id,
            items=items,
            priority_gap=parsed.get("priority_gap"),
            retrieval_sources=[],
        )


class NoRAGQuestionGenerator:
    """Template-based question generation without KB retrieval."""

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def generate_opening_question(
        self, concept_id: str
    ) -> tuple[str, list[str]]:
        messages = [
            SystemMessage(
                content=(
                    "You are a technical interviewer. Generate an interview question "
                    "about the given topic. Also list expected answer elements. "
                    "Respond in JSON: {\"question\": \"...\", \"expected_elements\": [...]}"
                )
            ),
            HumanMessage(content=f"Topic: {concept_id}"),
        ]
        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())
        return parsed["question"], parsed["expected_elements"]

    async def generate_followup_question(
        self, gap_report: GapReport, qa_history: list[dict[str, str]]
    ) -> tuple[str, list[str]]:
        messages = [
            SystemMessage(
                content=(
                    "You are a technical interviewer. Generate a follow-up question "
                    "based on gaps in the candidate's previous answer. "
                    "Respond in JSON: {\"question\": \"...\", \"expected_elements\": [...]}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {gap_report.concept_id}\n"
                    f"Gap: {gap_report.priority_gap}\n"
                    "Generate a follow-up question."
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())
        return parsed["question"], parsed["expected_elements"]


class NoRAGAssessor:
    """LLM-only assessment without retrieved ground truth."""

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def assess(
        self,
        candidate_response: str,
        expected_elements: list[str],
        concept_id: str,
        question_text: str,
    ) -> AssessmentRecord:
        expected_text = "\n".join(f"- {e}" for e in expected_elements)
        messages = [
            SystemMessage(
                content=(
                    "You are assessing a candidate's technical interview response. "
                    "Score each expected element from 0.0 to 1.0. "
                    "Respond in JSON: {\"element_scores\": [{\"element\": ..., "
                    "\"score\": ..., \"justification\": ..., \"grounding_source\": \"llm\"}], "
                    "\"overall_score\": ..., \"misconceptions_detected\": [...]}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {concept_id}\n"
                    f"Question: {question_text}\n"
                    f"Expected elements:\n{expected_text}\n\n"
                    f"Candidate response:\n{candidate_response}"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())
        element_scores = [
            ElementScore.model_validate(es) for es in parsed["element_scores"]
        ]
        return AssessmentRecord(
            concept_id=concept_id,
            question_text=question_text,
            response_text=candidate_response,
            element_scores=element_scores,
            overall_score=parsed["overall_score"],
            grounding_sources=[],
            misconceptions_detected=parsed.get("misconceptions_detected", []),
        )


class NoRAGFeedback:
    """LLM-only feedback synthesis without KB citations."""

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def synthesize(
        self,
        assessments: list[AssessmentRecord],
        topics_covered: list[TopicState],
    ) -> FeedbackReport:
        assessment_text = "\n".join(
            f"- {a.concept_id}: score={a.overall_score}" for a in assessments
        )
        messages = [
            SystemMessage(
                content=(
                    "Generate interview feedback based on assessment scores. "
                    "Respond in JSON: {\"overall_summary\": ..., \"topic_summaries\": "
                    "[{\"concept_name\": ..., \"score\": ..., \"strengths\": [...], "
                    "\"gaps\": [...], \"recommendations\": [...], \"cited_sources\": []}], "
                    "\"strengths\": [...], \"areas_for_improvement\": [...], "
                    "\"kb_references\": []}"
                )
            ),
            HumanMessage(content=f"Assessment records:\n{assessment_text}"),
        ]
        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())
        topic_summaries = [
            TopicFeedback.model_validate(ts) for ts in parsed["topic_summaries"]
        ]
        return FeedbackReport(
            overall_summary=parsed["overall_summary"],
            topic_summaries=topic_summaries,
            strengths=parsed["strengths"],
            areas_for_improvement=parsed["areas_for_improvement"],
            kb_references=[],
        )
