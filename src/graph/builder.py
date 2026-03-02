"""StateGraph construction, edge routing, and compilation."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.agents.dma import DialogueManagerAgent
from src.agents.fsa import FeedbackSynthesisAgent
from src.agents.kgda import KnowledgeGapDetectionAgent
from src.agents.qga import QuestionGenerationAgent
from src.agents.raa import ResponseAssessmentAgent
from src.config import Settings
from src.graph import nodes
from src.graph.state import InterviewState
from src.kb.retriever import KBRetriever


def route_by_action(state: InterviewState) -> str:
    """Route to the next node based on DMA action decision."""
    action = state.get("dma_action")
    if action is None:
        return "trigger_fsa"
    return action.value


def build_interview_graph(
    llm: BaseChatModel,
    retriever: KBRetriever,
    settings: Settings,
    variant: Literal["full", "no_rag", "no_kgda", "no_raa_ground"] = "full",
) -> StateGraph:
    """Build and compile the interview LangGraph.

    Graph flow:
      [candidate_input]
        -> classify_intent
        -> decide_action
        -> CONDITIONAL:
            delegate_kgda -> run_kgda -> run_raa -> decide_action (loop)
            delegate_qga  -> run_qga -> deliver_question -> END
            respond_directly -> END
            move_on -> run_qga -> deliver_question -> END
            trigger_fsa -> run_fsa -> END

    Args:
        llm: The LLM to use for agents.
        retriever: KB retriever instance.
        settings: Application settings.
        variant: System variant for baseline/ablation experiments.
    """
    # Inject agent instances into the nodes module
    dma = DialogueManagerAgent(llm, retriever, settings)
    nodes.dma_agent = dma
    nodes.fsa_agent = FeedbackSynthesisAgent(llm, retriever)

    if variant == "no_kgda":
        from src.baseline.no_kgda import RandomKGDA
        nodes.kgda_agent = RandomKGDA(llm, retriever)
    else:
        nodes.kgda_agent = KnowledgeGapDetectionAgent(llm, retriever)

    if variant == "no_raa_ground":
        from src.baseline.no_raa_ground import UngroundedAssessor
        nodes.raa_agent = UngroundedAssessor(llm, retriever)
    else:
        nodes.raa_agent = ResponseAssessmentAgent(llm, retriever)

    if variant == "no_rag":
        from src.baseline.no_rag_agents import (
            NoRAGFeedback,
            NoRAGKGDA,
            NoRAGQuestionGenerator,
            NoRAGAssessor,
        )
        nodes.kgda_agent = NoRAGKGDA(llm, retriever)
        nodes.qga_agent = NoRAGQuestionGenerator(llm, retriever)
        nodes.raa_agent = NoRAGAssessor(llm, retriever)
        nodes.fsa_agent = NoRAGFeedback(llm, retriever)
    else:
        nodes.qga_agent = QuestionGenerationAgent(llm, retriever)

    # Build graph
    graph = StateGraph(InterviewState)

    graph.add_node("classify_intent", nodes.dma_classify_intent)
    graph.add_node("decide_action", nodes.dma_decide_action)
    graph.add_node("run_kgda", nodes.run_kgda)
    graph.add_node("run_raa", nodes.run_raa)
    graph.add_node("run_qga", nodes.run_qga)
    graph.add_node("run_fsa", nodes.run_fsa)
    graph.add_node("respond_directly", nodes.dma_respond_directly)
    graph.add_node("move_on", nodes.dma_move_on)
    graph.add_node("deliver_question", nodes.dma_deliver_question)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "decide_action")

    graph.add_conditional_edges(
        "decide_action",
        route_by_action,
        {
            "delegate_kgda": "run_kgda",
            "delegate_qga": "run_qga",
            "respond_directly": "respond_directly",
            "move_on": "move_on",
            "trigger_fsa": "run_fsa",
        },
    )

    graph.add_edge("run_kgda", "run_raa")
    graph.add_edge("run_raa", "decide_action")  # Loop back for decision
    graph.add_edge("run_qga", "deliver_question")
    graph.add_edge("deliver_question", END)  # Wait for next candidate input
    graph.add_edge("respond_directly", END)  # Wait for next candidate input
    graph.add_edge("move_on", "run_qga")  # Generate opening Q for new topic
    graph.add_edge("run_fsa", END)  # Session complete

    return graph.compile()
