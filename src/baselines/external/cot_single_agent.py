"""External baseline: Chain-of-Thought Single Agent.

A single LLM with structured CoT prompting that simulates multi-step
reasoning (analyze gaps → decide action → generate question/feedback)
within one call per turn. No KB retrieval — relies on parametric knowledge.
More sophisticated than single_llm but still a single agent.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.baselines.external.external_base import ExternalBaseline, ExternalBaselineResult
from src.models import (
    AssessmentRecord,
    ElementScore,
    FeedbackReport,
    TopicFeedback,
)
from src.simulation.candidate import SyntheticCandidate

COT_SYSTEM_PROMPT = """\
You are a technical interviewer for Data Structures and Algorithms.
You must cover these topics: {topics}

For EACH turn, think step by step and output ONLY valid JSON (no markdown fences):

{{
  "reasoning": {{
    "step1_gap_analysis": "<What did the candidate get right/wrong in their last response?>",
    "step2_decision": "<Should I: ask a follow-up on the same topic, move to the next topic, or end the interview?>",
    "step3_justification": "<Why this decision?>"
  }},
  "action": "ask_question" | "move_on" | "end_interview",
  "current_topic": "<concept_id being discussed>",
  "question": "<Your question to the candidate (if action is ask_question or move_on)>",
  "assessment_of_last_response": {{
    "concept_id": "<topic of the last response>",
    "overall_score": <0.0-1.0>,
    "elements": [
      {{"element": "<aspect>", "score": <0.0-1.0>, "justification": "<reason>"}}
    ],
    "misconceptions": ["<any detected>"]
  }} | null
}}

When action is "end_interview", also include:
{{
  "action": "end_interview",
  "final_feedback": {{
    "overall_summary": "<2-3 sentence summary>",
    "topic_scores": [
      {{"concept_id": "<topic>", "score": <0.0-1.0>, "strengths": [...], "gaps": [...]}}
    ],
    "strengths": ["<overall strength>"],
    "areas_for_improvement": ["<area>"]
  }}
}}

RULES:
- Cover all topics within {max_turns} exchanges
- Ask at most 2 follow-up questions per topic before moving on
- Always assess the previous response before asking the next question
- Start by asking about the first topic"""


class CoTSingleAgentInterviewer(ExternalBaseline):
    """Chain-of-Thought single agent — structured reasoning, no RAG."""

    async def run_interview(
        self,
        topic_concept_ids: list[str],
        candidate: SyntheticCandidate,
        max_turns: int = 30,
    ) -> ExternalBaselineResult:
        topics_str = ", ".join(topic_concept_ids)
        messages: list = [
            SystemMessage(
                content=COT_SYSTEM_PROMPT.format(
                    topics=topics_str, max_turns=max_turns
                )
            ),
            HumanMessage(content="Begin the interview now. Output your first turn as JSON."),
        ]

        conversation: list[dict[str, str]] = []
        questions_asked: list[tuple[str, str]] = []
        assessments: list[AssessmentRecord] = []
        final_feedback_raw: dict | None = None

        for turn in range(max_turns):
            # LLM produces CoT + action as JSON
            response = await self.llm.ainvoke(messages)
            raw_text = response.content.strip()
            messages.append(AIMessage(content=raw_text))

            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract question from text
                conversation.append({"role": "interviewer", "content": raw_text})
                candidate_response = await candidate.respond(raw_text)
                messages.append(HumanMessage(content=candidate_response))
                conversation.append({"role": "candidate", "content": candidate_response})
                continue

            action = parsed.get("action", "ask_question")

            # Collect assessment of previous response
            assessment_data = parsed.get("assessment_of_last_response")
            if assessment_data and isinstance(assessment_data, dict):
                assessments.append(self._to_assessment_record(assessment_data))

            if action == "end_interview":
                final_feedback_raw = parsed.get("final_feedback")
                break

            # Extract question and present to candidate
            question = parsed.get("question", "")
            current_topic = parsed.get("current_topic", "unknown")
            if not question:
                continue

            questions_asked.append((current_topic, question))
            conversation.append({"role": "interviewer", "content": question})

            candidate_response = await candidate.respond(question)
            conversation.append({"role": "candidate", "content": candidate_response})
            messages.append(
                HumanMessage(
                    content=f"Candidate's response: {candidate_response}\n\n"
                    "Produce your next turn as JSON."
                )
            )

        # Force final assessment if loop ended without one
        if final_feedback_raw is None:
            final_feedback_raw = await self._force_final_feedback(
                messages, topic_concept_ids
            )

        feedback = self._to_feedback_report(final_feedback_raw or {}, assessments)

        return ExternalBaselineResult(
            assessments=assessments,
            feedback=feedback,
            questions_asked=questions_asked,
            conversation_history=conversation,
        )

    async def _force_final_feedback(
        self, messages: list, topic_concept_ids: list[str]
    ) -> dict | None:
        messages.append(
            HumanMessage(
                content=(
                    "The interview is over. Output your final turn with "
                    '"action": "end_interview" and "final_feedback" as JSON.'
                )
            )
        )
        response = await self.llm.ainvoke(messages)
        try:
            parsed = json.loads(response.content.strip())
            return parsed.get("final_feedback")
        except json.JSONDecodeError:
            return None

    def _to_assessment_record(self, data: dict) -> AssessmentRecord:
        elements = data.get("elements", [])
        element_scores = [
            ElementScore(
                element=e.get("element", ""),
                score=e.get("score", 0.5),
                justification=e.get("justification", ""),
                grounding_source="llm_cot",
            )
            for e in elements
        ]
        return AssessmentRecord(
            concept_id=data.get("concept_id", "unknown"),
            question_text="",
            response_text="",
            element_scores=element_scores,
            overall_score=data.get("overall_score", 0.5),
            grounding_sources=[],
            misconceptions_detected=data.get("misconceptions", []),
        )

    def _to_feedback_report(
        self, raw: dict, assessments: list[AssessmentRecord]
    ) -> FeedbackReport:
        topic_scores = raw.get("topic_scores", [])
        topic_summaries = [
            TopicFeedback(
                concept_name=ts.get("concept_id", "unknown"),
                score=ts.get("score", 0.5),
                strengths=ts.get("strengths", []),
                gaps=ts.get("gaps", []),
                recommendations=[],
                cited_sources=[],
            )
            for ts in topic_scores
        ]

        # Fallback: build from assessments if no topic_scores in feedback
        if not topic_summaries and assessments:
            topic_summaries = [
                TopicFeedback(
                    concept_name=a.concept_id,
                    score=a.overall_score,
                    strengths=[],
                    gaps=[],
                    recommendations=[],
                    cited_sources=[],
                )
                for a in assessments
            ]

        return FeedbackReport(
            overall_summary=raw.get("overall_summary", "Interview completed."),
            topic_summaries=topic_summaries,
            strengths=raw.get("strengths", []),
            areas_for_improvement=raw.get("areas_for_improvement", []),
            kb_references=[],
        )
