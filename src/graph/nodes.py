"""LangGraph node functions. Each takes InterviewState and returns a partial state update."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.dma import DialogueManagerAgent
from src.agents.fsa import FeedbackSynthesisAgent
from src.agents.kgda import KnowledgeGapDetectionAgent
from src.agents.qga import QuestionGenerationAgent
from src.agents.raa import ResponseAssessmentAgent
from src.graph.state import InterviewState
from src.models import DMAAction, IntentType


# These module-level variables are set by the graph builder to inject agent instances.
dma_agent: DialogueManagerAgent | None = None
kgda_agent: KnowledgeGapDetectionAgent | None = None
qga_agent: QuestionGenerationAgent | None = None
raa_agent: ResponseAssessmentAgent | None = None
fsa_agent: FeedbackSynthesisAgent | None = None


def _get_last_human_message(state: InterviewState) -> str:
    """Extract text of the last human message."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _get_last_question(state: InterviewState) -> str:
    """Extract the last question asked (from pending_question or topic state)."""
    if state.get("pending_question"):
        return state["pending_question"]
    topic = state.get("current_topic")
    if topic and topic.questions_asked:
        return topic.questions_asked[-1]
    return ""


async def dma_classify_intent(state: InterviewState) -> dict:
    """Classify the latest candidate message. Updates classified_intent."""
    assert dma_agent is not None
    candidate_msg = _get_last_human_message(state)
    question = _get_last_question(state)
    intent = await dma_agent.classify_intent(candidate_msg, question)
    return {
        "classified_intent": intent,
        "session_turn_count": state["session_turn_count"] + 1,
    }


async def dma_decide_action(state: InterviewState) -> dict:
    """Decide DMA action based on intent + state. Updates dma_action."""
    assert dma_agent is not None
    topic = state["current_topic"]
    intent = state["classified_intent"] or IntentType.SUBSTANTIVE_RESPONSE

    # Check if session should end
    if dma_agent.should_end_session(
        topic,
        state["topic_queue"],
        state["session_turn_count"],
        state["max_turns"],
    ):
        return {"dma_action": DMAAction.TRIGGER_FSA}

    # If no current topic, end session
    if topic is None:
        return {"dma_action": DMAAction.TRIGGER_FSA}

    # Handle non-substantive intents
    if intent in (IntentType.CLARIFICATION_QUESTION, IntentType.REQUEST_HINT):
        return {"dma_action": DMAAction.RESPOND_DIRECTLY}
    if intent == IntentType.NON_ANSWER:
        if topic.depth >= dma_agent.settings.max_depth_per_topic:
            return {"dma_action": DMAAction.MOVE_ON}
        return {"dma_action": DMAAction.RESPOND_DIRECTLY}

    # For substantive responses, use LLM to decide
    gap_summary = "No gap report yet"
    gap_report = state.get("current_gap_report")
    if gap_report:
        gap_items = [
            f"{item.property_text}: {item.status}" for item in gap_report.items
        ]
        gap_summary = "; ".join(gap_items)
        if gap_report.priority_gap:
            gap_summary += f" | Priority gap: {gap_report.priority_gap}"

    action = await dma_agent.decide_action(
        topic_name=topic.concept_name,
        depth=topic.depth,
        max_depth=dma_agent.settings.max_depth_per_topic,
        topics_remaining=len(state["topic_queue"]),
        turns_used=state["session_turn_count"],
        max_turns=state["max_turns"],
        intent=intent,
        gap_summary=gap_summary,
    )
    return {"dma_action": action}


async def run_kgda(state: InterviewState) -> dict:
    """Run gap analysis on latest response. Updates current_gap_report."""
    assert kgda_agent is not None
    topic = state["current_topic"]
    if topic is None:
        return {}

    candidate_response = _get_last_human_message(state)
    gap_report = await kgda_agent.analyze(candidate_response, topic.concept_id)

    # Update topic state
    topic.gap_reports.append(gap_report)

    return {"current_gap_report": gap_report, "current_topic": topic}


async def run_raa(state: InterviewState) -> dict:
    """Score latest response. Updates current_assessment + all_assessments."""
    assert raa_agent is not None
    topic = state["current_topic"]
    if topic is None:
        return {}

    candidate_response = _get_last_human_message(state)
    question_text = _get_last_question(state)
    expected_elements = state.get("expected_elements", [])

    assessment = await raa_agent.assess(
        candidate_response=candidate_response,
        expected_elements=expected_elements,
        concept_id=topic.concept_id,
        question_text=question_text,
    )

    # Update topic state
    topic.assessments.append(assessment)

    all_assessments = list(state.get("all_assessments", []))
    all_assessments.append(assessment)

    return {
        "current_assessment": assessment,
        "all_assessments": all_assessments,
        "current_topic": topic,
    }


async def run_qga(state: InterviewState) -> dict:
    """Generate next question. Updates pending_question + expected_elements."""
    assert qga_agent is not None
    topic = state["current_topic"]
    if topic is None:
        return {}

    gap_report = state.get("current_gap_report")

    if not topic.questions_asked or gap_report is None:
        # Opening question
        question, elements = await qga_agent.generate_opening_question(
            topic.concept_id
        )
    else:
        # Follow-up question
        qa_history = [
            {"question": q, "response": a.response_text}
            for q, a in zip(topic.questions_asked, topic.assessments)
        ]
        question, elements = await qga_agent.generate_followup_question(
            gap_report, qa_history
        )

    # Update topic state
    topic.questions_asked.append(question)
    topic.depth += 1

    return {
        "pending_question": question,
        "expected_elements": elements,
        "current_topic": topic,
    }


async def run_fsa(state: InterviewState) -> dict:
    """Generate final feedback. Updates feedback_report + session_complete."""
    assert fsa_agent is not None

    all_topics = list(state.get("topics_covered", []))
    if state.get("current_topic"):
        current = state["current_topic"]
        current.status = "completed"
        all_topics.append(current)

    feedback = await fsa_agent.synthesize(
        assessments=state.get("all_assessments", []),
        topics_covered=all_topics,
    )

    return {
        "feedback_report": feedback,
        "session_complete": True,
    }


async def dma_respond_directly(state: InterviewState) -> dict:
    """Generate clarification/hint. Updates messages."""
    assert dma_agent is not None
    topic = state["current_topic"]
    if topic is None:
        return {}

    candidate_msg = _get_last_human_message(state)
    question = _get_last_question(state)
    intent = state["classified_intent"] or IntentType.NON_ANSWER

    response_text = await dma_agent.generate_direct_response(
        topic_name=topic.concept_name,
        question=question,
        candidate_message=candidate_msg,
        intent=intent,
    )

    return {"messages": [AIMessage(content=response_text)]}


async def dma_move_on(state: InterviewState) -> dict:
    """Advance to next topic. Updates current_topic, topic_queue, topics_covered."""
    assert dma_agent is not None
    topic = state["current_topic"]
    if topic is None:
        return {"session_complete": True}

    update = dma_agent.advance_topic(topic, list(state["topic_queue"]))

    # Convert topics_covered to a list append
    covered = list(state.get("topics_covered", []))
    if isinstance(update.get("topics_covered"), list):
        covered.extend(update["topics_covered"])
    elif update.get("topics_covered") is not None:
        covered.append(update["topics_covered"])
    update["topics_covered"] = covered

    return update


async def dma_deliver_question(state: InterviewState) -> dict:
    """Package pending_question into messages for candidate delivery."""
    question = state.get("pending_question", "")
    if question:
        return {"messages": [AIMessage(content=question)]}
    return {}
