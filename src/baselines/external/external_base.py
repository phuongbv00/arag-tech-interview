"""Abstract base class and shared types for external baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from src.models import AssessmentRecord, FeedbackReport
from src.simulation.candidate import SyntheticCandidate


@dataclass
class ExternalBaselineResult:
    """Standardized output from any external baseline interview."""

    assessments: list[AssessmentRecord] = field(default_factory=list)
    feedback: FeedbackReport | None = None
    questions_asked: list[tuple[str, str]] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)


class ExternalBaseline(ABC):
    """Interface that all external baselines must implement.

    External baselines bypass the LangGraph pipeline entirely — they run
    their own interview loop and produce results in the same format for
    metric comparison.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    @abstractmethod
    async def run_interview(
        self,
        topic_concept_ids: list[str],
        candidate: SyntheticCandidate,
        max_turns: int = 30,
    ) -> ExternalBaselineResult:
        """Run a complete interview session and return structured results."""
        ...
