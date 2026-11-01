"""One output-only language model; no access to evidence or hidden memories."""

from copy import deepcopy
import json
from pathlib import Path

from core.interfaces import GenerationResult, LanguageModel
from patient_agents.interaction import response_schema, validate_interaction
from patient_agents.structured import generate_object


DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "patient_agents" / "response.txt"
REPAIR_PROMPT = DEFAULT_PROMPT.with_name("format_repair.txt")


class ResponseAgent:
    def __init__(self, llm: LanguageModel, response_profile: dict, system_prompt: str = "",
                 prompt: str | None = None, repair_prompt: str | None = None,
                 interaction_enabled: bool = True, format_prompt: str | None = None):
        self.llm = llm
        self.response_profile = deepcopy(response_profile)
        self.prompt = DEFAULT_PROMPT.read_text(encoding="utf-8").strip() if prompt is None else prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("The response prompt must be nonempty text.")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text.")
        if type(interaction_enabled) is not bool:
            raise ValueError("interaction_enabled must be a boolean.")
        self.interaction_enabled = interaction_enabled
        format_path = DEFAULT_PROMPT.with_name("interaction.txt" if interaction_enabled else "utterance.txt")
        self.format_prompt = format_path.read_text(encoding="utf-8").strip() if format_prompt is None else format_prompt
        if not isinstance(self.format_prompt, str) or not self.format_prompt.strip():
            raise ValueError("The response format prompt must be nonempty text.")
        self.system_prompt = "\n\n".join(part for part in (self.prompt, system_prompt, self.format_prompt) if part.strip())
        self.repair_prompt = REPAIR_PROMPT.read_text(encoding="utf-8").strip() if repair_prompt is None else repair_prompt
        if not isinstance(self.repair_prompt, str) or not self.repair_prompt.strip():
            raise ValueError("The format repair prompt must be nonempty text.")
        self.model_input: list[dict[str, str]] = []
        self.last_trace = {}
        self.interaction = None

    def respond(self, turn_context: dict) -> GenerationResult:
        # Put earlier material first and the current therapist stimulus last.
        # This changes presentation only: no extra history or hidden units.
        context = deepcopy(turn_context)
        if not self.interaction_enabled:
            context.get("interaction_state", {}).pop("previous_exchange", None)
        current = context.pop("accessible_current_units")
        context["accessible_current_units"] = current
        payload = {"patient": self.response_profile, "turn_context": context}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ]
        self.model_input = deepcopy(messages)
        self.last_trace = {"interaction_enabled": self.interaction_enabled}
        self.interaction = None

        def validate(value):
            fields = {"utterance", "interaction"} if self.interaction_enabled else {"utterance"}
            if (set(value) != fields
                    or not isinstance(value["utterance"], str) or not value["utterance"].strip()):
                raise ValueError(f"Patient JSON requires exactly {sorted(fields)} and a nonempty utterance.")
            parsed = {"utterance": value["utterance"].strip()}
            if self.interaction_enabled:
                parsed["interaction"] = validate_interaction(value["interaction"], context)
            return parsed

        try:
            parsed, result = generate_object(self.llm, messages, validate, self.repair_prompt, self.last_trace,
                                             json_schema=response_schema(context, self.interaction_enabled))
        finally:
            attempts = self.last_trace.get("generation_attempts", [])
            if attempts:
                self.model_input = deepcopy(attempts[-1]["model_input"])
        self.interaction = parsed.get("interaction")
        if self.interaction_enabled:
            self.last_trace["interaction"] = deepcopy(self.interaction)
        return {**result, "content": parsed["utterance"]}
