from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import GapItem, GapReport
from src.prompts.kgda_prompts import GAP_ANALYSIS_SYSTEM, GAP_ANALYSIS_USER


class KnowledgeGapDetectionAgent:
    """Retrieves expected concept content, analyzes candidate response for gaps.

    Trigger: response quality (not candidate query) — detects omissions
    the candidate would not flag themselves.
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def analyze(
        self,
        candidate_response: str,
        concept_id: str,
    ) -> GapReport:
        """Analyze a candidate response for knowledge gaps.

        Steps:
        1. Retrieve key_properties + example_correct_response for concept_id
        2. LLM compares candidate response against retrieved properties
        3. Classify each property status
        4. Identify priority_gap
        5. Return GapReport with retrieval_sources
        """
        concept = self.retriever.get_by_concept_id(concept_id)
        if concept is None:
            return GapReport(
                concept_id=concept_id,
                concept_name=concept_id,
                items=[],
                priority_gap=None,
                retrieval_sources=[],
            )

        key_properties_text = "\n".join(
            f"- {prop}" for prop in concept.key_properties
        )

        messages = [
            SystemMessage(content=GAP_ANALYSIS_SYSTEM),
            HumanMessage(
                content=GAP_ANALYSIS_USER.format(
                    concept_name=concept.concept_name,
                    concept_id=concept.concept_id,
                    key_properties=key_properties_text,
                    example_correct_response=concept.example_correct_response,
                    candidate_response=candidate_response,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())

        items = [GapItem.model_validate(item) for item in parsed["items"]]

        return GapReport(
            concept_id=concept.concept_id,
            concept_name=concept.concept_name,
            items=items,
            priority_gap=parsed.get("priority_gap"),
            retrieval_sources=[concept.concept_id],
        )
