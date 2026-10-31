"""Small shared contracts for language models and patient methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


@dataclass(frozen=True)
class DialogueMessage:
    speaker: Literal["Therapist", "Patient"]
    content: str


class GenerationResult(TypedDict):
    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: Literal["eos", "length"]


class LanguageModel(Protocol):
    def reply(self, messages: list[dict[str, str]]) -> GenerationResult:
        """Generate one reply from model-facing chat messages."""
        ...


class Patient(Protocol):
    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        """Return one visible patient reply from the public dialogue."""
        ...
