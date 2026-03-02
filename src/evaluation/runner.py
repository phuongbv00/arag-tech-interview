"""Orchestrates full evaluation: run sessions, collect results, score."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import chromadb
from anthropic import AsyncAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.config import Settings
from src.evaluation.metrics import (
    element_precision_recall,
    evidence_citation_rate,
    gap_coverage_rate,
    hallucination_rate,
)
from src.evaluation.ragas_adapter import SessionData, compute_ragas_metrics, format_for_ragas
from src.graph.builder import build_interview_graph
from src.kb.loader import load_concepts
from src.kb.retriever import KBRetriever
from src.simulation.candidate import SyntheticCandidate
from src.simulation.personas import (
    PersonaProfile,
    build_l1_persona,
    build_l3_persona,
)

logger = logging.getLogger(__name__)

Variant = Literal["full", "no_rag", "no_kgda", "no_raa_ground"]


@dataclass
class SessionResult:
    """Results from a single interview session."""

    variant: str
    persona_level: str
    repetition: int
    assessments: list[dict] = field(default_factory=list)
    feedback: dict = field(default_factory=dict)
    custom_metrics: dict[str, float] = field(default_factory=dict)
    ragas_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Aggregated results from all evaluation sessions."""

    sessions: list[SessionResult] = field(default_factory=list)


class EvaluationRunner:
    """Orchestrates the full evaluation pipeline."""

    def __init__(
        self,
        settings: Settings,
        kb_path: Path,
        candidate_model: str = "claude-3-5-sonnet-20241022",
        repetitions: int = 3,
    ) -> None:
        self.settings = settings
        self.kb_path = kb_path
        self.candidate_model = candidate_model
        self.repetitions = repetitions

        # Initialize clients
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

        # Load KB
        self.concepts = load_concepts(kb_path)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        self.retriever = KBRetriever(
            self.collection, self.openai_client, settings.embedding_model
        )

        # LLM for agents
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    def _build_persona(self, level: str) -> PersonaProfile:
        if level == "L1":
            return build_l1_persona(self.concepts)
        return build_l3_persona(self.concepts)

    async def run_session(
        self,
        variant: Variant,
        persona: PersonaProfile,
    ) -> SessionResult:
        """Run a single interview session with a synthetic candidate."""
        # Build the graph for this variant
        graph = build_interview_graph(
            llm=self.llm,
            retriever=self.retriever,
            settings=self.settings,
            variant=variant,
        )

        # Build the synthetic candidate
        candidate = SyntheticCandidate(
            profile=persona,
            anthropic_client=self.anthropic_client,
            model=self.candidate_model,
        )

        # Initialize session
        from src.agents.dma import DialogueManagerAgent

        dma = DialogueManagerAgent(self.llm, self.retriever, self.settings)
        init_state = dma.initialize_session()
        state = {
            **init_state,
            "messages": [],
            "classified_intent": None,
            "dma_action": None,
            "dma_direct_response": None,
            "current_gap_report": None,
            "current_assessment": None,
            "pending_question": None,
            "expected_elements": [],
        }

        # Generate first question via QGA
        from src.agents.qga import QuestionGenerationAgent
        from src.prompts.dma_prompts import OPENING_MESSAGE

        from langchain_core.messages import AIMessage, HumanMessage

        # Add opening message
        state["messages"] = [AIMessage(content=OPENING_MESSAGE)]

        # Run the interview loop
        max_iterations = self.settings.max_total_turns * 2
        for _ in range(max_iterations):
            if state.get("session_complete"):
                break

            # If there's a pending question, deliver it and get candidate response
            if state.get("pending_question"):
                question = state["pending_question"]
                candidate_response = await candidate.respond(question)
                state["messages"].append(HumanMessage(content=candidate_response))

                # Run the graph with the new candidate input
                result = await graph.ainvoke(state)
                state.update(result)
            else:
                # Need to generate first question - run QGA directly
                topic = state.get("current_topic")
                if topic:
                    qga = QuestionGenerationAgent(self.llm, self.retriever)
                    question, elements = await qga.generate_opening_question(
                        topic.concept_id
                    )
                    state["pending_question"] = question
                    state["expected_elements"] = elements
                    topic.questions_asked.append(question)
                    topic.depth += 1
                else:
                    break

        # Collect results
        result = SessionResult(
            variant=variant,
            persona_level=persona.level,
            repetition=0,
        )

        # Store assessments
        for a in state.get("all_assessments", []):
            result.assessments.append(a.model_dump())

        # Store feedback
        feedback = state.get("feedback_report")
        if feedback:
            result.feedback = feedback.model_dump()

        # Compute custom metrics
        planted_gaps = persona.partial_concepts + persona.unknown_concepts
        detected_gaps = [
            a.concept_id
            for a in state.get("all_assessments", [])
            if a.overall_score < 0.5
        ]
        result.custom_metrics["gap_coverage_rate"] = gap_coverage_rate(
            detected_gaps, planted_gaps
        )

        precision, recall = element_precision_recall(
            state.get("all_assessments", []), persona
        )
        result.custom_metrics["element_precision"] = precision
        result.custom_metrics["element_recall"] = recall

        kb_ids = {c.concept_id for c in self.concepts}
        result.custom_metrics["hallucination_rate"] = hallucination_rate(
            state.get("all_assessments", []), kb_ids
        )

        if feedback:
            result.custom_metrics["evidence_citation_rate"] = evidence_citation_rate(
                feedback
            )

        return result

    async def run_full_evaluation(self) -> EvaluationResults:
        """Run all experiment combinations.

        Matrix: 4 variants x 2 persona levels x N repetitions.
        """
        variants: list[Variant] = ["full", "no_rag", "no_kgda", "no_raa_ground"]
        persona_levels = ["L1", "L3"]
        results = EvaluationResults()

        for variant in variants:
            for level in persona_levels:
                for rep in range(self.repetitions):
                    logger.info(
                        "Running: variant=%s, level=%s, rep=%d",
                        variant, level, rep,
                    )
                    persona = self._build_persona(level)
                    session_result = await self.run_session(variant, persona)
                    session_result.repetition = rep
                    results.sessions.append(session_result)

        return results
