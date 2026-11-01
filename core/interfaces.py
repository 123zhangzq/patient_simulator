"""Small shared contracts for language models and patient methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict


@dataclass(frozen=True)
class DialogueMessage:
    speaker: Literal["Therapist", "Patient"]
    content: str


class GenerationResult(TypedDict):
    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: Literal["eos", "length"]
    dialogue_action: NotRequired[Literal["continue", "clarify", "close"]]
    stop_reason: NotRequired[str | None]


class LanguageModel(Protocol):
    def reply(self, messages: list[dict[str, str]]) -> GenerationResult:
        """Generate one reply from model-facing chat messages."""
        ...


class Patient(Protocol):
    last_trace: dict | None

    def initialize_history(self, history: Sequence[DialogueMessage]) -> None:
        """Initialize the supplied public prefix without generating new speech."""
        ...

    def describe(self) -> dict:
        """Return an audit snapshot; this is never part of the public dialogue."""
        ...

    def respond(self, public_history: Sequence[DialogueMessage]) -> GenerationResult:
        """Return one visible patient reply from the public dialogue."""
        ...
