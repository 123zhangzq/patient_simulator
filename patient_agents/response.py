"""One output-only language model; no access to evidence or hidden memories."""

from copy import deepcopy
import json
from pathlib import Path

from core.interfaces import GenerationResult, LanguageModel


DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "patient_agents" / "response.txt"


class ResponseAgent:
    def __init__(self, llm: LanguageModel, response_profile: dict, system_prompt: str = "",
                 prompt: str | None = None):
        self.llm = llm
        self.response_profile = deepcopy(response_profile)
        self.prompt = DEFAULT_PROMPT.read_text(encoding="utf-8").strip() if prompt is None else prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("The response prompt must be nonempty text.")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text.")
        self.system_prompt = "\n\n".join(part for part in (self.prompt, system_prompt) if part.strip())
        self.model_input: list[dict[str, str]] = []

    def respond(self, turn_context: dict) -> GenerationResult:
        payload = {"patient": self.response_profile, "turn_context": turn_context}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ]
        self.model_input = deepcopy(messages)
        return self.llm.reply(messages)
