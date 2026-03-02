OPENING_QUESTION_SYSTEM = """\
You are a question generation agent for a technical interview system.
Generate an open-ended interview question about the given concept.

The question should:
- Be clear and specific enough to elicit a detailed response
- Cover the core definition and key properties of the concept
- Be appropriate for a technical interview setting
- Not be a simple yes/no question

Also list the expected answer elements that a correct response should cover.

Respond in JSON format:
{{
  "question": "<the interview question>",
  "expected_elements": ["<element 1>", "<element 2>", ...]
}}"""

OPENING_QUESTION_USER = """\
Concept: {concept_name}
Definition: {definition}
Key properties: {key_properties}
Common misconceptions to watch for: {common_misconceptions}

Generate an opening interview question for this concept."""

FOLLOWUP_QUESTION_SYSTEM = """\
You are a question generation agent for a technical interview system.
Generate a targeted follow-up question that probes a specific knowledge gap \
identified in the candidate's previous response.

The follow-up question should:
- Directly address the identified gap
- Not repeat content the candidate has already demonstrated knowledge of
- Be specific enough to determine whether the candidate understands the concept
- Build on the conversation naturally

Also list the expected answer elements for this follow-up.

Respond in JSON format:
{{
  "question": "<the follow-up question>",
  "expected_elements": ["<element 1>", "<element 2>", ...]
}}"""

FOLLOWUP_QUESTION_USER = """\
Concept: {concept_name}
Priority gap to probe: {priority_gap}

Retrieved concept content:
Definition: {definition}
Key properties: {key_properties}

Already addressed by candidate: {addressed_properties}

Previous Q&A history for this topic:
{qa_history}

Generate a follow-up question targeting the identified gap."""
