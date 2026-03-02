"""External baseline: Static Question Bank (No Adaptation).

Pre-defined questions asked in fixed order regardless of candidate responses.
Represents template-based approaches like HireVue and Pathak & Pandey (2025).
No gap analysis, no follow-up adaptation.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.baselines.external.external_base import ExternalBaseline, ExternalBaselineResult
from src.kb.retriever import KBRetriever
from src.models import (
    AssessmentRecord,
    ElementScore,
    FeedbackReport,
    TopicFeedback,
)
from src.simulation.candidate import SyntheticCandidate

QUESTION_TEMPLATE = "Can you explain {concept_name}? What are its key properties and characteristics?"

ASSESS_SYSTEM = """\
You are assessing a candidate's technical interview response.
Score the response from 0.0 to 1.0 based on correctness and completeness.
Identify strengths, gaps, and any misconceptions.

Respond in JSON (no markdown fences):
{{
  "overall_score": <0.0-1.0>,
  "element_scores": [
    {{
      "element": "<aspect evaluated>",
      "score": <0.0-1.0>,
      "justification": "<brief reason>",
      "grounding_source": "llm_static_bank"
    }}
  ],
  "misconceptions_detected": ["<misconception or empty>"],
  "strengths": ["<strength>"],
  "gaps": ["<gap>"]
}}"""

ASSESS_USER = """\
Topic: {concept_name}
Question: {question}
Candidate's response: {response}

Assess this response."""

FEEDBACK_SYSTEM = """\
Generate a brief interview feedback summary based on per-topic scores.
Respond in JSON (no markdown fences):
{{
  "overall_summary": "<2-3 sentence summary>",
  "strengths": ["<strength>"],
  "areas_for_improvement": ["<area>"]
}}"""


class StaticQuestionBankInterviewer(ExternalBaseline):
    """Fixed question bank — one pre-defined question per topic, no adaptation.

    Questions are generated from concept names at init time.
    Assessment uses LLM without KB retrieval (to isolate the adaptation variable).
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever | None = None) -> None:
        super().__init__(llm)
        self.retriever = retriever

    async def run_interview(
        self,
        topic_concept_ids: list[str],
        candidate: SyntheticCandidate,
        max_turns: int = 30,
    ) -> ExternalBaselineResult:
        assessments: list[AssessmentRecord] = []
        questions_asked: list[tuple[str, str]] = []
        conversation: list[dict[str, str]] = []

        for concept_id in topic_concept_ids:
            concept_name = concept_id.replace("dsa-", "").replace("-", " ").title()

            # Use KB for concept name if available
            if self.retriever:
                concept = self.retriever.get_by_concept_id(concept_id)
                if concept:
                    concept_name = concept.concept_name

            # Fixed question — no adaptation
            question = QUESTION_TEMPLATE.format(concept_name=concept_name)
            questions_asked.append((concept_id, question))
            conversation.append({"role": "interviewer", "content": question})

            # Get candidate response
            candidate_response = await candidate.respond(question)
            conversation.append({"role": "candidate", "content": candidate_response})

            # Assess with LLM (no KB retrieval)
            assessment = await self._assess_response(
                concept_id, concept_name, question, candidate_response
            )
            assessments.append(assessment)

        # Generate overall feedback
        feedback = await self._generate_feedback(assessments)

        return ExternalBaselineResult(
            assessments=assessments,
            feedback=feedback,
            questions_asked=questions_asked,
            conversation_history=conversation,
        )

    async def _assess_response(
        self,
        concept_id: str,
        concept_name: str,
        question: str,
        response: str,
    ) -> AssessmentRecord:
        messages = [
            SystemMessage(content=ASSESS_SYSTEM),
            HumanMessage(
                content=ASSESS_USER.format(
                    concept_name=concept_name,
                    question=question,
                    response=response,
                )
            ),
        ]

        llm_response = await self.llm.ainvoke(messages)
        try:
            parsed = json.loads(llm_response.content.strip())
        except json.JSONDecodeError:
            parsed = {"overall_score": 0.5, "element_scores": [], "misconceptions_detected": []}

        element_scores = [
            ElementScore.model_validate(es) for es in parsed.get("element_scores", [])
        ]

        return AssessmentRecord(
            concept_id=concept_id,
            question_text=question,
            response_text=response,
            element_scores=element_scores,
            overall_score=parsed.get("overall_score", 0.5),
            grounding_sources=[],
            misconceptions_detected=parsed.get("misconceptions_detected", []),
        )

    async def _generate_feedback(
        self, assessments: list[AssessmentRecord]
    ) -> FeedbackReport:
        scores_text = "\n".join(
            f"- {a.concept_id}: {a.overall_score:.2f}" for a in assessments
        )
        messages = [
            SystemMessage(content=FEEDBACK_SYSTEM),
            HumanMessage(content=f"Per-topic scores:\n{scores_text}"),
        ]

        response = await self.llm.ainvoke(messages)
        try:
            parsed = json.loads(response.content.strip())
        except json.JSONDecodeError:
            parsed = {
                "overall_summary": "Interview completed.",
                "strengths": [],
                "areas_for_improvement": [],
            }

        topic_summaries = [
            TopicFeedback(
                concept_name=a.concept_id,
                score=a.overall_score,
                strengths=[],
                gaps=[],
                recommendations=[],
                cited_sources=[],
            )
            for a in assessments
        ]

        return FeedbackReport(
            overall_summary=parsed.get("overall_summary", "Interview completed."),
            topic_summaries=topic_summaries,
            strengths=parsed.get("strengths", []),
            areas_for_improvement=parsed.get("areas_for_improvement", []),
            kb_references=[],
        )
