from __future__ import annotations

import random

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Settings
from src.kb.retriever import KBRetriever
from src.models import DMAAction, IntentType, TopicState
from src.prompts.dma_prompts import (
    ACTION_DECISION_SYSTEM,
    ACTION_DECISION_USER,
    DIRECT_RESPONSE_SYSTEM,
    DIRECT_RESPONSE_USER,
    INTENT_CLASSIFICATION_SYSTEM,
    INTENT_CLASSIFICATION_USER,
)


class DialogueManagerAgent:
    """Hybrid agent: deterministic state management + LLM conversational reasoning.

    The DMA is the sole interface with the candidate. It manages session state
    deterministically and uses LLM for intent classification, action decisions,
    and direct responses (clarifications, hints).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: KBRetriever,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings

    # --- Deterministic methods (no LLM) ---

    def initialize_session(
        self,
        topic_concept_ids: list[str] | None = None,
    ) -> dict:
        """Select initial topics and populate the session state.

        If topic_concept_ids is not provided, selects a balanced set from the KB.
        Returns partial state update for InterviewState.
        """
        if topic_concept_ids is None:
            topic_concept_ids = self._select_topics()

        first_id = topic_concept_ids[0]
        concept = self.retriever.get_by_concept_id(first_id)
        current_topic = TopicState(
            concept_id=first_id,
            concept_name=concept.concept_name if concept else first_id,
        )

        return {
            "current_topic": current_topic,
            "topic_queue": topic_concept_ids[1:],
            "topics_covered": [],
            "session_turn_count": 0,
            "max_turns": self.settings.max_total_turns,
            "all_assessments": [],
            "session_complete": False,
            "feedback_report": None,
        }

    def _select_topics(self) -> list[str]:
        """Select a balanced set of topics across difficulty levels."""
        max_topics = self.settings.max_topics
        all_concepts = self.retriever.get_all_concepts()

        by_difficulty: dict[str, list[str]] = {
            "beginner": [],
            "intermediate": [],
            "advanced": [],
        }
        for c in all_concepts:
            by_difficulty[c.difficulty_level].append(c.concept_id)

        for ids in by_difficulty.values():
            random.shuffle(ids)

        selected: list[str] = []
        # Distribute: ~40% beginner, ~40% intermediate, ~20% advanced
        counts = {
            "beginner": max(1, int(max_topics * 0.4)),
            "intermediate": max(1, int(max_topics * 0.4)),
            "advanced": max(1, max_topics - max(1, int(max_topics * 0.4)) * 2),
        }
        for level, count in counts.items():
            selected.extend(by_difficulty[level][:count])

        return selected[:max_topics]

    def advance_topic(self, current_topic: TopicState, topic_queue: list[str]) -> dict:
        """Mark current topic as completed and advance to the next.

        Returns partial state update.
        """
        current_topic.status = "completed"
        topics_covered_update = current_topic

        if not topic_queue:
            return {
                "current_topic": None,
                "topic_queue": [],
                "topics_covered": topics_covered_update,
            }

        next_id = topic_queue[0]
        concept = self.retriever.get_by_concept_id(next_id)
        next_topic = TopicState(
            concept_id=next_id,
            concept_name=concept.concept_name if concept else next_id,
        )

        return {
            "current_topic": next_topic,
            "topic_queue": topic_queue[1:],
            "topics_covered": topics_covered_update,
        }

    def should_end_session(
        self,
        current_topic: TopicState | None,
        topic_queue: list[str],
        session_turn_count: int,
        max_turns: int,
    ) -> bool:
        """Check whether the session should end."""
        if session_turn_count >= max_turns:
            return True
        if current_topic is None and not topic_queue:
            return True
        return False

    # --- LLM methods ---

    async def classify_intent(
        self,
        candidate_message: str,
        current_question: str,
    ) -> IntentType:
        """Classify the candidate's last message into an IntentType."""
        messages = [
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM),
            HumanMessage(
                content=INTENT_CLASSIFICATION_USER.format(
                    question=current_question,
                    candidate_message=candidate_message,
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        intent_str = response.content.strip().lower()

        for intent in IntentType:
            if intent.value == intent_str:
                return intent
        return IntentType.SUBSTANTIVE_RESPONSE

    async def decide_action(
        self,
        topic_name: str,
        depth: int,
        max_depth: int,
        topics_remaining: int,
        turns_used: int,
        max_turns: int,
        intent: IntentType,
        gap_summary: str,
    ) -> DMAAction:
        """Decide the next action based on interview state."""
        messages = [
            SystemMessage(content=ACTION_DECISION_SYSTEM),
            HumanMessage(
                content=ACTION_DECISION_USER.format(
                    topic_name=topic_name,
                    depth=depth,
                    max_depth=max_depth,
                    topics_remaining=topics_remaining,
                    turns_used=turns_used,
                    max_turns=max_turns,
                    intent=intent.value,
                    gap_summary=gap_summary,
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        action_str = response.content.strip().lower()

        for action in DMAAction:
            if action.value == action_str:
                return action
        return DMAAction.DELEGATE_KGDA

    async def generate_direct_response(
        self,
        topic_name: str,
        question: str,
        candidate_message: str,
        intent: IntentType,
    ) -> str:
        """Generate a clarification, hint, or acknowledgment."""
        messages = [
            SystemMessage(content=DIRECT_RESPONSE_SYSTEM),
            HumanMessage(
                content=DIRECT_RESPONSE_USER.format(
                    topic_name=topic_name,
                    question=question,
                    candidate_message=candidate_message,
                    intent=intent.value,
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()
