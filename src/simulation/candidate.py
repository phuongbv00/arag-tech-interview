"""Claude 3.5 Sonnet candidate simulator for synthetic interview evaluation."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from src.simulation.personas import PersonaProfile


def _build_system_prompt(profile: PersonaProfile) -> str:
    """Build the system prompt encoding the persona's knowledge profile."""
    mastered_text = ", ".join(profile.mastered_concepts) or "None"
    partial_text = ", ".join(profile.partial_concepts) or "None"
    unknown_text = ", ".join(profile.unknown_concepts) or "None"

    misconception_lines: list[str] = []
    for cid, misconceptions in profile.misconception_map.items():
        for m in misconceptions:
            misconception_lines.append(f"  - {cid}: {m}")
    misconception_text = "\n".join(misconception_lines) or "  None"

    style = (
        "shorter, less precise, sometimes uncertain"
        if profile.level == "L1"
        else "detailed, technical, confident"
    )

    return f"""\
You are a synthetic technical interview candidate at proficiency level {profile.level}.
You are being interviewed about Data Structures and Algorithms.

YOUR KNOWLEDGE PROFILE:
- Concepts you FULLY KNOW (give correct, complete answers): {mastered_text}
- Concepts you PARTIALLY KNOW (know definition but miss some key properties): {partial_text}
- Concepts you DON'T KNOW (say "I'm not sure" or give vague/incorrect answers): {unknown_text}

MISCONCEPTIONS TO EXHIBIT (when relevant):
{misconception_text}

RESPONSE STYLE:
- Your responses should be {style}
- For mastered concepts: give thorough, accurate answers
- For partial concepts: give partially correct answers, miss some key details
- For unknown concepts: say "I'm not really sure about that" or attempt an answer \
that includes misconceptions listed above
- Stay in character — do not reveal you are an AI or mention your knowledge profile
- Keep responses to 2-5 sentences unless the question requires more detail"""


class SyntheticCandidate:
    """Uses Claude 3.5 Sonnet to simulate candidate responses."""

    def __init__(
        self,
        profile: PersonaProfile,
        anthropic_client: AsyncAnthropic,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        self.profile = profile
        self.client = anthropic_client
        self.model = model
        self.system_prompt = _build_system_prompt(profile)
        self.conversation_history: list[dict[str, str]] = []

    async def respond(self, interviewer_message: str) -> str:
        """Generate a response consistent with the persona profile."""
        self.conversation_history.append(
            {"role": "user", "content": interviewer_message}
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.conversation_history,
        )

        assistant_text = response.content[0].text
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        return assistant_text
