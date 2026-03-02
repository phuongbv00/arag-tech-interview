"""Ablation: LLM-only scoring without retrieved ground truth.

Isolates the contribution of RAG-grounded assessment (RQ2).
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import AssessmentRecord, ElementScore


class UngroundedAssessor:
    """Replaces RAA: LLM scores response using only parametric knowledge.

    No KB retrieval is performed. The LLM assesses correctness based on
    its own training knowledge rather than retrieved ground truth.
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever  # kept for interface compatibility

    async def assess(
        self,
        candidate_response: str,
        expected_elements: list[str],
        concept_id: str,
        question_text: str,
    ) -> AssessmentRecord:
        """Score response using LLM parametric knowledge only (no retrieval)."""
        expected_text = "\n".join(f"- {e}" for e in expected_elements)
        messages = [
            SystemMessage(
                content=(
                    "You are assessing a candidate's technical interview response "
                    "about a data structures and algorithms topic. "
                    "Score each expected element from 0.0 to 1.0 based on your "
                    "knowledge of the topic. Check for misconceptions. "
                    "Respond in JSON: {\"element_scores\": [{\"element\": ..., "
                    "\"score\": ..., \"justification\": ..., "
                    "\"grounding_source\": \"llm_parametric\"}], "
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
            grounding_sources=[],  # No KB sources used
            misconceptions_detected=parsed.get("misconceptions_detected", []),
        )
