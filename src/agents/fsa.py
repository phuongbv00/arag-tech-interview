from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.retriever import KBRetriever
from src.models import AssessmentRecord, FeedbackReport, TopicFeedback, TopicState
from src.prompts.fsa_prompts import FEEDBACK_SYNTHESIS_SYSTEM, FEEDBACK_SYNTHESIS_USER


class FeedbackSynthesisAgent:
    """Generates end-of-session feedback grounded in KB references.

    All feedback statements must cite specific concept_ids from the
    knowledge base. Triggered once at session end by the DMA.
    """

    def __init__(self, llm: BaseChatModel, retriever: KBRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def synthesize(
        self,
        assessments: list[AssessmentRecord],
        topics_covered: list[TopicState],
    ) -> FeedbackReport:
        """Generate a comprehensive feedback report.

        Steps:
        1. Aggregate all assessments by topic
        2. For each gap, retrieve cited KB content
        3. LLM generates per-topic summary with strengths + gaps + recommendations
        4. Compile overall summary
        5. All statements cite specific concept_ids
        """
        # Collect all referenced concept_ids
        all_concept_ids: set[str] = set()
        for assessment in assessments:
            all_concept_ids.update(assessment.grounding_sources)
        for topic in topics_covered:
            all_concept_ids.add(topic.concept_id)

        # Retrieve KB content for all cited concepts
        kb_content_parts: list[str] = []
        for cid in sorted(all_concept_ids):
            concept = self.retriever.get_by_concept_id(cid)
            if concept:
                kb_content_parts.append(
                    f"[{concept.concept_id}] {concept.concept_name}:\n"
                    f"  Definition: {concept.definition}\n"
                    f"  Key properties: {'; '.join(concept.key_properties)}"
                )
        kb_content_text = "\n\n".join(kb_content_parts) or "No KB content available"

        # Format assessment records
        assessment_text_parts: list[str] = []
        for a in assessments:
            scores_text = ", ".join(
                f"{es.element}: {es.score}" for es in a.element_scores
            )
            assessment_text_parts.append(
                f"Topic: {a.concept_id}\n"
                f"  Question: {a.question_text}\n"
                f"  Overall score: {a.overall_score}\n"
                f"  Element scores: {scores_text}\n"
                f"  Misconceptions: {', '.join(a.misconceptions_detected) or 'None'}"
            )
        assessment_text = "\n\n".join(assessment_text_parts) or "No assessments"

        # Format topics covered
        topics_text_parts: list[str] = []
        for t in topics_covered:
            topics_text_parts.append(
                f"- {t.concept_name} ({t.concept_id}): "
                f"depth={t.depth}, status={t.status}, "
                f"questions={len(t.questions_asked)}"
            )
        topics_text = "\n".join(topics_text_parts) or "No topics"

        messages = [
            SystemMessage(content=FEEDBACK_SYNTHESIS_SYSTEM),
            HumanMessage(
                content=FEEDBACK_SYNTHESIS_USER.format(
                    assessment_records=assessment_text,
                    topics_covered=topics_text,
                    kb_content=kb_content_text,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = json.loads(response.content.strip())

        topic_summaries = [
            TopicFeedback.model_validate(ts) for ts in parsed["topic_summaries"]
        ]

        return FeedbackReport(
            overall_summary=parsed["overall_summary"],
            topic_summaries=topic_summaries,
            strengths=parsed["strengths"],
            areas_for_improvement=parsed["areas_for_improvement"],
            kb_references=parsed.get("kb_references", []),
        )
