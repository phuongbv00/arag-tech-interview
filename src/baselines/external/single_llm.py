"""External baseline: Single-LLM Interviewer.

One LLM prompt handles the entire interview — question generation,
follow-up, assessment, and feedback — with no agents, no RAG, no
structured state. Represents the simplest possible AI interview system.
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

SYSTEM_PROMPT = """\
You are a technical interviewer conducting a Data Structures and Algorithms interview.

Your responsibilities:
- Ask clear, open-ended technical questions about the given topics
- Listen to the candidate's response and ask follow-up questions when needed
- Decide when to move to the next topic
- At the end, provide a summary assessment

Topics to cover (in order): {topics}

RULES:
- Ask ONE question at a time
- After the candidate responds, either ask a follow-up or move to the next topic
- Cover all topics within {max_turns} total exchanges
- When you've covered all topics (or are running low on turns), output ONLY \
the JSON assessment below — no other text.

When done, output EXACTLY this JSON (no markdown fences):
{{
  "done": true,
  "assessments": [
    {{
      "concept_id": "<topic>",
      "question_text": "<the question you asked>",
      "overall_score": <0.0-1.0>,
      "strengths": ["<strength>"],
      "gaps": ["<gap>"],
      "misconceptions": ["<misconception or empty>"]
    }}
  ],
  "overall_summary": "<2-3 sentence summary>",
  "strengths": ["<strength>"],
  "areas_for_improvement": ["<area>"]
}}

Until you are done, just respond naturally as an interviewer."""

NEXT_TURN_PROMPT = """\
Candidate's response: {response}

Either ask a follow-up question, move to the next topic, or if all topics \
are covered output the final JSON assessment."""


class SingleLLMInterviewer(ExternalBaseline):
    """Single-LLM interviewer — no agents, no RAG, no structured state."""

    async def run_interview(
        self,
        topic_concept_ids: list[str],
        candidate: SyntheticCandidate,
        max_turns: int = 30,
    ) -> ExternalBaselineResult:
        topics_str = ", ".join(topic_concept_ids)
        messages: list = [
            SystemMessage(
                content=SYSTEM_PROMPT.format(topics=topics_str, max_turns=max_turns)
            ),
            HumanMessage(content="Please begin the interview."),
        ]

        conversation: list[dict[str, str]] = []
        questions_asked: list[tuple[str, str]] = []
        raw_assessments: list[dict] = []

        for turn in range(max_turns):
            # LLM generates interviewer turn
            response = await self.llm.ainvoke(messages)
            interviewer_text = response.content.strip()
            messages.append(AIMessage(content=interviewer_text))
            conversation.append({"role": "interviewer", "content": interviewer_text})

            # Check if LLM output the final JSON assessment
            if self._is_final_assessment(interviewer_text):
                raw_assessments = self._parse_final_assessment(interviewer_text)
                break

            # Get candidate response
            candidate_response = await candidate.respond(interviewer_text)
            messages.append(
                HumanMessage(
                    content=NEXT_TURN_PROMPT.format(response=candidate_response)
                )
            )
            conversation.append({"role": "candidate", "content": candidate_response})

            # Track questions (best-effort topic assignment by order)
            topic_idx = min(turn // 2, len(topic_concept_ids) - 1)
            questions_asked.append(
                (topic_concept_ids[topic_idx], interviewer_text)
            )

        # If loop ended without final assessment, force one
        if not raw_assessments:
            raw_assessments = await self._force_final_assessment(
                messages, topic_concept_ids
            )

        # Convert to standardized types
        assessments = self._to_assessment_records(raw_assessments)
        feedback = self._to_feedback_report(raw_assessments)

        return ExternalBaselineResult(
            assessments=assessments,
            feedback=feedback,
            questions_asked=questions_asked,
            conversation_history=conversation,
        )

    def _is_final_assessment(self, text: str) -> bool:
        try:
            data = json.loads(text)
            return data.get("done") is True
        except (json.JSONDecodeError, AttributeError):
            return False

    def _parse_final_assessment(self, text: str) -> list[dict]:
        try:
            data = json.loads(text)
            return data.get("assessments", [])
        except (json.JSONDecodeError, AttributeError):
            return []

    async def _force_final_assessment(
        self, messages: list, topic_concept_ids: list[str]
    ) -> list[dict]:
        messages.append(
            HumanMessage(
                content=(
                    "The interview is over. Output ONLY the final JSON assessment "
                    "now, covering all topics discussed."
                )
            )
        )
        response = await self.llm.ainvoke(messages)
        return self._parse_final_assessment(response.content.strip())

    def _to_assessment_records(
        self, raw: list[dict]
    ) -> list[AssessmentRecord]:
        records: list[AssessmentRecord] = []
        for item in raw:
            strengths = item.get("strengths", [])
            gaps = item.get("gaps", [])
            all_elements = strengths + gaps
            element_scores = [
                ElementScore(
                    element=s,
                    score=1.0,
                    justification="Identified as strength",
                    grounding_source="llm_single",
                )
                for s in strengths
            ] + [
                ElementScore(
                    element=g,
                    score=0.0,
                    justification="Identified as gap",
                    grounding_source="llm_single",
                )
                for g in gaps
            ]
            records.append(
                AssessmentRecord(
                    concept_id=item.get("concept_id", "unknown"),
                    question_text=item.get("question_text", ""),
                    response_text="",
                    element_scores=element_scores,
                    overall_score=item.get("overall_score", 0.5),
                    grounding_sources=[],
                    misconceptions_detected=item.get("misconceptions", []),
                )
            )
        return records

    def _to_feedback_report(self, raw: list[dict]) -> FeedbackReport:
        topic_summaries = [
            TopicFeedback(
                concept_name=item.get("concept_id", "unknown"),
                score=item.get("overall_score", 0.5),
                strengths=item.get("strengths", []),
                gaps=item.get("gaps", []),
                recommendations=[],
                cited_sources=[],
            )
            for item in raw
        ]

        # Extract overall summary from the last raw entry if available
        overall_summary = ""
        all_strengths: list[str] = []
        all_areas: list[str] = []
        if raw:
            # These fields come from the top-level JSON, try to find them
            for item in raw:
                all_strengths.extend(item.get("strengths", []))
                all_areas.extend(item.get("gaps", []))

        return FeedbackReport(
            overall_summary=overall_summary or "Interview completed.",
            topic_summaries=topic_summaries,
            strengths=all_strengths,
            areas_for_improvement=all_areas,
            kb_references=[],
        )
