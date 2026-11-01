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
                 prompt: str | None = None, repair_prompt: str | None = None):
        self.llm = llm
        self.response_profile = deepcopy(response_profile)
        self.prompt = DEFAULT_PROMPT.read_text(encoding="utf-8").strip() if prompt is None else prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("The response prompt must be nonempty text.")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text.")
        self.system_prompt = "\n\n".join(part for part in (self.prompt, system_prompt) if part.strip())
        self.repair_prompt = REPAIR_PROMPT.read_text(encoding="utf-8").strip() if repair_prompt is None else repair_prompt
        if not isinstance(self.repair_prompt, str) or not self.repair_prompt.strip():
            raise ValueError("The format repair prompt must be nonempty text.")
        self.model_input: list[dict[str, str]] = []
        self.last_trace = {}
        self.interaction = None

    def respond(self, turn_context: dict) -> GenerationResult:
        payload = {"patient": self.response_profile, "turn_context": turn_context}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ]
        self.model_input = deepcopy(messages)
        self.last_trace = {}
        self.interaction = None

        def validate(value):
            if (set(value) != {"utterance", "interaction"}
                    or not isinstance(value["utterance"], str) or not value["utterance"].strip()):
                raise ValueError("Patient JSON requires a nonempty utterance and interaction object.")
            return {"utterance": value["utterance"].strip(),
                    "interaction": validate_interaction(value["interaction"], turn_context)}

        try:
            parsed, result = generate_object(self.llm, messages, validate, self.repair_prompt, self.last_trace,
                                             json_schema=response_schema(turn_context))
        finally:
            attempts = self.last_trace.get("generation_attempts", [])
            if attempts:
                self.model_input = deepcopy(attempts[-1]["model_input"])
        self.interaction = parsed["interaction"]
        self.last_trace["interaction"] = deepcopy(self.interaction)
        return {**result, "content": parsed["utterance"]}
