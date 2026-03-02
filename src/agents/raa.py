from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import AssessmentRecord, ElementScore
from src.prompts.raa_prompts import ASSESSMENT_SYSTEM, ASSESSMENT_USER


class ResponseAssessmentAgent:
    """Scores candidate responses against retrieved ground truth.

    Key property: correctness is judged against retrieved KB reference,
    not LLM parametric memory. Each score includes a grounding_source
    for full auditability.
    """

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
        """Score a candidate response against expected elements and retrieved ground truth.

        Steps:
        1. Retrieve example_correct_response + common_misconceptions
        2. LLM scores each expected_element (0.0-1.0) with justification
        3. Check for misconception matches
        4. Each score includes grounding_source (concept_id)
        5. Return AssessmentRecord
        """
        concept = self.retriever.get_by_concept_id(concept_id)
        if concept is None:
            return AssessmentRecord(
                concept_id=concept_id,
                question_text=question_text,
                response_text=candidate_response,
                element_scores=[],
                overall_score=0.0,
                grounding_sources=[],
                misconceptions_detected=[],
            )

        expected_text = "\n".join(f"- {e}" for e in expected_elements)
        misconceptions_text = "\n".join(
            f"- {m}" for m in concept.common_misconceptions
        ) or "None documented"

        messages = [
            SystemMessage(content=ASSESSMENT_SYSTEM),
            HumanMessage(
                content=ASSESSMENT_USER.format(
                    concept_name=concept.concept_name,
                    concept_id=concept.concept_id,
                    question_text=question_text,
                    expected_elements=expected_text,
                    example_correct_response=concept.example_correct_response,
                    common_misconceptions=misconceptions_text,
                    candidate_response=candidate_response,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())

        element_scores = [
            ElementScore.model_validate(es) for es in parsed["element_scores"]
        ]

        return AssessmentRecord(
            concept_id=concept.concept_id,
            question_text=question_text,
            response_text=candidate_response,
            element_scores=element_scores,
            overall_score=parsed["overall_score"],
            grounding_sources=[concept.concept_id],
            misconceptions_detected=parsed.get("misconceptions_detected", []),
        )
