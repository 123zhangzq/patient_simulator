"""A patient baseline that makes one model call per visible reply."""

from collections.abc import Sequence
import copy

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel
from core.profiles import profile_fingerprint, render_patient_profile


class SinglePromptPatient:
    def __init__(self, llm: LanguageModel, system_prompt: str, *, patient_profile: dict,
                 seed: int = 42, options: dict | None = None, prompts: dict | None = None):
        self.llm = llm
        self.patient_profile = copy.deepcopy(patient_profile)
        self.system_prompt = system_prompt + "\n\nCase patient information (JSON)\n" + render_patient_profile(patient_profile)
        self.last_trace = None
        self.initial_history_size = 0

    def initialize_history(self, history: Sequence[DialogueMessage]) -> None:
        # The same prefix is kept in public_history for every later model call.
        self.initial_history_size = len(history)

    def describe(self) -> dict:
        return {"method": "single_prompt", "profile_sha256": profile_fingerprint(self.patient_profile),
                "initial_history_utterances": self.initial_history_size}

    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(
            {
                "role": "assistant" if message.speaker == "Patient" else "user",
                "content": message.content,
            }
            for message in public_history
        )
        self.last_trace = {"method": "single_prompt", "model_input": copy.deepcopy(messages)}
        return self.llm.reply(messages)
