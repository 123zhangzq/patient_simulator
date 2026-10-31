"""A patient baseline that makes one model call per visible reply."""

from collections.abc import Sequence

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel


class SinglePromptPatient:
    def __init__(self, llm: LanguageModel, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(
            {
                "role": "assistant" if message.speaker == "Patient" else "user",
                "content": message.content,
            }
            for message in public_history
        )
        return self.llm.reply(messages)
