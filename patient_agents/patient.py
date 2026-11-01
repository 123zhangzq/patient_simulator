"""Compose the three responsibilities behind the shared patient interface."""

from collections.abc import Sequence
from copy import deepcopy
import hashlib

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel
from patient_agents.context import ContextController
from patient_agents.response import ResponseAgent
from patient_agents.static import compile_profile


def _digest(message: DialogueMessage) -> str:
    return hashlib.sha256(f"{message.speaker}\0{message.content}".encode("utf-8")).hexdigest()


class PatientAgents:
    def __init__(self, llm: LanguageModel, system_prompt: str, *, patient_profile: dict,
                 seed: int = 42, options: dict | None = None, prompts: dict | None = None):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer from 0 to 2**32-1.")
        self.options = deepcopy({} if options is None else options)
        self.compiled = compile_profile(patient_profile, self.options)
        prompts = {} if prompts is None else prompts
        if not isinstance(prompts, dict) or prompts.keys() - {"response"}:
            raise ValueError("prompts supports only a response text override.")
        self.context = ContextController(self.compiled, seed,
                                         attention_enabled=self.options.get("attention_enabled", True),
                                         memory_enabled=self.options.get("memory_enabled", True))
        self.response = ResponseAgent(llm, self.compiled["response_profile"], system_prompt,
                                      prompt=prompts.get("response"))
        self.seed = seed
        self.last_trace: dict = {}
        self._history_digests: list[str] = []
        self._history_initialized = False
        self._initialization = {"message_count": 0, "therapist_turns": 0, "processed_turns": []}

    def initialize_history(self, history: Sequence[DialogueMessage]) -> None:
        """Experience the explicitly supplied prefix once, without model calls.

        Both methods receive the same public prefix. The proposed method passes
        its therapist stimuli through its usual attention/encoding mechanism;
        supplied patient turns record prior self-reports, not new generations.
        """
        if self._history_initialized or self._history_digests or self.context.turn:
            raise ValueError("Initial history can only be supplied once before the first response.")
        prefix = tuple(history)
        if len(prefix) % 2:
            raise ValueError("Initial history must contain complete Therapist/Patient pairs.")
        for index, message in enumerate(prefix):
            expected = "Therapist" if index % 2 == 0 else "Patient"
            if (not isinstance(message, DialogueMessage) or message.speaker != expected
                    or not isinstance(message.content, str) or not message.content.strip()):
                raise ValueError("Initial history must alternate nonempty Therapist and Patient turns.")
        # Validate the entire prefix before mutating its cognitive state.
        self._history_initialized = True
        traces = []
        for message in prefix:
            if message.speaker == "Therapist":
                self.context.advance(message.content)
                traces.append(deepcopy(self.context.last_trace))
            else:
                identifiers = self.context.record_patient_reply(message.content, from_initial_history=True)
                traces[-1]["patient_utterance_memory_ids"] = identifiers
                traces[-1]["patient_reply_stored_as_fact"] = False
        self._history_digests = [_digest(message) for message in prefix]
        self._initialization = {"message_count": len(prefix), "therapist_turns": self.context.turn,
                                "processed_turns": traces}

    def describe(self) -> dict:
        return deepcopy({"method": "patient_agents", "seed": self.seed,
                         "profile_sha256": self.compiled["profile_fingerprint"],
                         "attention_enabled": self.context.attention_enabled,
                         "memory_enabled": self.context.memory_enabled,
                         "compiled_profile": self.compiled,
                         "initial_history": self._initialization,
                         "response_prompt": self.response.system_prompt})

    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        hashes = [_digest(message) for message in public_history]
        previous = len(self._history_digests)
        if hashes[:previous] != self._history_digests or len(hashes) < previous:
            raise ValueError("Patient history must be append-only; create a new patient to start again.")
        new = public_history[previous:]
        therapist_messages = [message for message in new if message.speaker == "Therapist"]
        if not therapist_messages or not public_history or public_history[-1].speaker != "Therapist":
            raise ValueError("Each patient response requires a new therapist utterance at the end.")
        traces = []
        for message in therapist_messages:
            turn_context = self.context.advance(message.content)
            traces.append(deepcopy(self.context.last_trace))
        # Consume new inputs once even if generation fails. A failed generation
        # stops the run; retrying cannot silently encode the same input twice.
        self._history_digests = hashes
        self.last_trace = {**traces[-1], "processed_therapist_turns": traces}
        try:
            result = self.response.respond(turn_context)
        finally:
            self.last_trace["model_input"] = deepcopy(self.response.model_input)
        # Only an explicitly typed memory of speaking is stored; a generated
        # claim never becomes a personal fact or verified real-world episode.
        self.last_trace["patient_utterance_memory_ids"] = self.context.record_patient_reply(result["content"])
        self.last_trace["patient_reply_stored_as_fact"] = False
        return result
