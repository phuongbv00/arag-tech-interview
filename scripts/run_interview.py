"""CLI: Interactive interview session for manual testing."""

import asyncio

import chromadb
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.agents.dma import DialogueManagerAgent
from src.agents.qga import QuestionGenerationAgent
from src.config import Settings
from src.graph.builder import build_interview_graph
from src.kb.retriever import KBRetriever
from src.prompts.dma_prompts import OPENING_MESSAGE


async def main() -> None:
    settings = Settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)

    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name
    )
    retriever = KBRetriever(collection, openai_client, settings.embedding_model)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
    )

    graph = build_interview_graph(llm, retriever, settings)

    dma = DialogueManagerAgent(llm, retriever, settings)
    state = dma.initialize_session()
    state.update(
        {
            "messages": [],
            "classified_intent": None,
            "dma_action": None,
            "dma_direct_response": None,
            "current_gap_report": None,
            "current_assessment": None,
            "pending_question": None,
            "expected_elements": [],
        }
    )

    print(OPENING_MESSAGE)
    print()

    # Generate first question
    topic = state["current_topic"]
    if topic:
        qga = QuestionGenerationAgent(llm, retriever)
        question, elements = await qga.generate_opening_question(topic.concept_id)
        state["pending_question"] = question
        state["expected_elements"] = elements
        topic.questions_asked.append(question)
        topic.depth += 1
        state["messages"] = [AIMessage(content=question)]
        print(f"Interviewer: {question}")
        print()

    while not state.get("session_complete"):
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Session ended by user.")
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = await graph.ainvoke(state)
        state.update(result)

        # Print the last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nInterviewer: {msg.content}\n")
                break

    # Print feedback if available
    feedback = state.get("feedback_report")
    if feedback:
        print("\n" + "=" * 60)
        print("FEEDBACK REPORT")
        print("=" * 60)
        print(f"\n{feedback.overall_summary}\n")
        for ts in feedback.topic_summaries:
            print(f"--- {ts.concept_name} (Score: {ts.score}) ---")
            for s in ts.strengths:
                print(f"  + {s}")
            for g in ts.gaps:
                print(f"  - {g}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
