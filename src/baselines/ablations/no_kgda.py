"""Ablation: random concept selection instead of gap-driven retrieval.

Isolates the contribution of KGDA gap detection (RQ1).
"""

from __future__ import annotations

import random

from langchain_core.language_models import BaseChatModel

from src.kb.retriever import KBRetriever
from src.models import GapItem, GapReport


class RandomKGDA:
    """Replaces KGDA: generates a gap report with random gap selection
    instead of analyzing the candidate's actual response against KB content.
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def analyze(self, candidate_response: str, concept_id: str) -> GapReport:
        """Return a gap report with randomly selected priority gap."""
        concept = self.retriever.get_by_concept_id(concept_id)
        if concept is None:
            return GapReport(
                concept_id=concept_id,
                concept_name=concept_id,
                items=[],
                priority_gap=None,
                retrieval_sources=[],
            )

        # Mark all properties as not_addressed (no actual analysis)
        items = [
            GapItem(
                property_text=prop,
                status="not_addressed",
                evidence="",
            )
            for prop in concept.key_properties
        ]

        # Randomly select a priority gap
        priority_gap = random.choice(concept.key_properties) if concept.key_properties else None

        return GapReport(
            concept_id=concept.concept_id,
            concept_name=concept.concept_name,
            items=items,
            priority_gap=priority_gap,
            retrieval_sources=[concept.concept_id],
        )
