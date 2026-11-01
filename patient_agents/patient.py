"""Compose the three responsibilities behind the shared patient interface."""

from collections.abc import Sequence
from copy import deepcopy
import hashlib

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel
from patient_agents.context import ContextController
from patient_agents.response import ResponseAgent
from patient_agents.static import compile_profile
from patient_agents.person_card import PersonCardBuilder
from patient_agents.interaction import InteractionTracker


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
        if not isinstance(prompts, dict) or prompts.keys() - {"response", "person_card", "format_repair"}:
            raise ValueError("prompts supports response, person_card and format_repair overrides.")
        self.context = ContextController(self.compiled, seed,
                                         attention_enabled=self.options.get("attention_enabled", True),
                                         memory_enabled=self.options.get("memory_enabled", True))
        self.response = ResponseAgent(llm, self.compiled["response_profile"], system_prompt,
                                      prompt=prompts.get("response"), repair_prompt=prompts.get("format_repair"))
        self.card_builder = PersonCardBuilder(
            llm, self.compiled, prompt=prompts.get("person_card"), repair_prompt=prompts.get("format_repair"),
            max_new_tokens=self.options.get("person_card_max_new_tokens", 768))
        self.person_card = {"tendencies": []}
        self.interaction = InteractionTracker()
        self.seed = seed
        self.last_trace: dict = {}
        self._history_digests: list[str] = []
        self._history_initialized = False
        self._initialization_complete = False
        self._initialization = {"message_count": 0, "therapist_turns": 0, "processed_turns": []}

    def initialize_history(self, history: Sequence[DialogueMessage]) -> None:
        """Build the stable card once, then experience the supplied prefix.

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
        self._initialization["message_count"] = len(prefix)
        if self.options.get("person_card_enabled", True):
            self.person_card = self.card_builder.build(prefix)
        else:
            self.card_builder.last_trace = {"status": "disabled"}
        self.response.response_profile["session_tendencies"] = deepcopy(self.person_card["tendencies"])
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
        self._initialization_complete = True

    def describe(self) -> dict:
        return deepcopy({"method": "patient_agents", "seed": self.seed,
                         "profile_sha256": self.compiled["profile_fingerprint"],
                         "attention_enabled": self.context.attention_enabled,
                         "memory_enabled": self.context.memory_enabled,
                         "compiled_profile": self.compiled,
                         "person_card": self.person_card,
                         "person_card_construction": self.card_builder.last_trace,
                         "initial_history": self._initialization,
                         "response_prompt": self.response.system_prompt,
                         "person_card_prompt": self.card_builder.prompt,
                         "format_repair_prompt": self.response.repair_prompt})

    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        if not self._history_initialized:
            self.initialize_history(())
        if not self._initialization_complete:
            raise ValueError("Patient initialization failed; create a new patient before responding.")
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
        previous_exchange = self.interaction.expose(turn_context)
        if previous_exchange is not None:
            turn_context["interaction_state"]["previous_exchange"] = previous_exchange
        traces[-1]["turn_context"] = deepcopy(turn_context)
        # Consume new inputs once even if generation fails. A failed generation
        # stops the run; retrying cannot silently encode the same input twice.
        self._history_digests = hashes
        self.last_trace = {**traces[-1], "processed_therapist_turns": traces}
        try:
            result = self.response.respond(turn_context)
        finally:
            self.last_trace["model_input"] = deepcopy(self.response.model_input)
            self.last_trace["response_generation"] = deepcopy(self.response.last_trace)
        # Only an explicitly typed memory of speaking is stored; a generated
        # claim never becomes a personal fact or verified real-world episode.
        self.last_trace["patient_utterance_memory_ids"] = self.context.record_patient_reply(result["content"])
        self.last_trace["patient_reply_stored_as_fact"] = False
        self.interaction.record(self.response.interaction, self.last_trace["patient_utterance_memory_ids"])
        self.last_trace["interaction_after_reply"] = deepcopy(self.interaction.previous)
        return result
