"""The fixed, single-prompt therapist used by every patient method."""

from collections.abc import Sequence

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel


class Therapist:
    def __init__(
        self,
        llm: LanguageModel,
        system_prompt: str,
        opening_prompt: str,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.opening_prompt = opening_prompt

    def respond(self, history: Sequence[DialogueMessage]) -> GenerationResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.opening_prompt},
        ]
        messages.extend(
            {
                "role": "assistant" if message.speaker == "Therapist" else "user",
                "content": message.content,
            }
            for message in history
        )
        return self.llm.reply(messages)
