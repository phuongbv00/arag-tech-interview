from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import GapReport
from src.prompts.qga_prompts import (
    FOLLOWUP_QUESTION_SYSTEM,
    FOLLOWUP_QUESTION_USER,
    OPENING_QUESTION_SYSTEM,
    OPENING_QUESTION_USER,
)


class QuestionGenerationAgent:
    """Generates grounded interview questions from KB content.

    For opening questions: retrieves concept definition and key properties.
    For follow-ups: targets specific gaps identified by KGDA.
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def generate_opening_question(
        self, concept_id: str
    ) -> tuple[str, list[str]]:
        """Generate an opening question for a new topic.

        Returns (question_text, expected_elements).
        """
        concept = self.retriever.get_by_concept_id(concept_id)
        if concept is None:
            return (
                f"Can you explain {concept_id}?",
                [],
            )

        key_properties_text = "\n".join(
            f"- {prop}" for prop in concept.key_properties
        )
        misconceptions_text = "\n".join(
            f"- {m}" for m in concept.common_misconceptions
        ) or "None documented"

        messages = [
            SystemMessage(content=OPENING_QUESTION_SYSTEM),
            HumanMessage(
                content=OPENING_QUESTION_USER.format(
                    concept_name=concept.concept_name,
                    definition=concept.definition,
                    key_properties=key_properties_text,
                    common_misconceptions=misconceptions_text,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())

        return parsed["question"], parsed["expected_elements"]

    async def generate_followup_question(
        self,
        gap_report: GapReport,
        qa_history: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        """Generate a targeted follow-up question based on identified gaps.

        Returns (question_text, expected_elements).
        """
        concept = self.retriever.get_by_concept_id(gap_report.concept_id)
        if concept is None:
            return (
                f"Can you elaborate more on {gap_report.priority_gap}?",
                [],
            )

        key_properties_text = "\n".join(
            f"- {prop}" for prop in concept.key_properties
        )

        addressed = [
            item.property_text
            for item in gap_report.items
            if item.status == "addressed_correctly"
        ]
        addressed_text = "\n".join(f"- {a}" for a in addressed) or "None yet"

        qa_text = ""
        for qa in qa_history:
            qa_text += f"Q: {qa.get('question', '')}\nA: {qa.get('response', '')}\n\n"
        qa_text = qa_text.strip() or "No previous Q&A"

        messages = [
            SystemMessage(content=FOLLOWUP_QUESTION_SYSTEM),
            HumanMessage(
                content=FOLLOWUP_QUESTION_USER.format(
                    concept_name=concept.concept_name,
                    priority_gap=gap_report.priority_gap or "general understanding",
                    definition=concept.definition,
                    key_properties=key_properties_text,
                    addressed_properties=addressed_text,
                    qa_history=qa_text,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())

        return parsed["question"], parsed["expected_elements"]
