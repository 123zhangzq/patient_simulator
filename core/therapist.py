"""One therapist model with explicit continuation, repair, and closing actions."""

from collections.abc import Sequence
from copy import deepcopy
import json
from pathlib import Path
import re

from core.interfaces import DialogueMessage, GenerationResult, LanguageModel


DEFAULT_CONTROL_PATH = Path(__file__).resolve().parents[1] / "prompts" / "therapist_control.json"
_CONTROL_FIELDS = {
    "response_format", "first_turn", "following_turn", "repair_limit", "last_turn",
    "stall_closing", "budget_closing", "format_repair",
}


def load_control(path: str | Path = DEFAULT_CONTROL_PATH) -> dict[str, str]:
    """Load editable dialogue-control wording before loading model weights."""
    control = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(control, dict) or set(control) != _CONTROL_FIELDS
            or any(not isinstance(value, str) or not value.strip() for value in control.values())):
        raise ValueError(f"Therapist control must contain nonempty text for {sorted(_CONTROL_FIELDS)}.")
    return control


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_reply(text: str) -> dict[str, str]:
    """Accept one complete object, optionally surrounded by prose or a fence.

    Do not guess between multiple objects, repair syntax locally, or extract an
    inner object from a different schema. Those cases need a model repair.
    """
    if not isinstance(text, str):
        raise ValueError("Therapist reply must be JSON text.")
    text = text.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_fields)
    try:
        try:
            reply = decoder.decode(text)
        except json.JSONDecodeError:
            start = re.search(r"[\[{]", text)
            if start is None:
                raise ValueError("No JSON object found.")
            reply, end = decoder.raw_decode(text, start.start())
            if re.search(r"[{}\[\]]", text[end:]):
                raise ValueError("Multiple or ambiguous JSON objects.")
    except ValueError:
        raise ValueError(
            "Therapist returned invalid JSON; expected one unambiguous object with utterance and action. "
            "Inspect therapist_trace.jsonl and the therapist control prompt."
        ) from None
    if (not isinstance(reply, dict) or set(reply) != {"utterance", "action"}
            or not isinstance(reply["utterance"], str) or not reply["utterance"].strip()
            or not isinstance(reply["action"], str)
            or reply["action"] not in {"continue", "clarify", "close"}):
        raise ValueError(
            "Therapist JSON requires a nonempty utterance and action continue, clarify, or close. "
            "Inspect therapist_trace.jsonl and the therapist control prompt."
        )
    return {"utterance": reply["utterance"].strip(), "action": reply["action"]}


def _question_key(text: str) -> str | None:
    """Recognize literal question repetition, not semantic paraphrases."""
    if "?" not in text and "？" not in text:
        return None
    return " ".join(re.sub(r"[^\w\s]|_", "", text.casefold()).split()) or None


class Therapist:
    def __init__(
        self,
        llm: LanguageModel,
        system_prompt: str,
        opening_prompt: str,
        *,
        max_clarifications: int = 2,
        max_rounds: int | None = None,
        opening_utterance: str | None = None,
        control_path: str | Path = DEFAULT_CONTROL_PATH,
    ):
        if type(max_clarifications) is not int or max_clarifications < 1:
            raise ValueError("max_clarifications must be a positive integer.")
        if max_rounds is not None and (type(max_rounds) is not int or max_rounds < 1):
            raise ValueError("max_rounds must be a positive integer or None.")
        if opening_utterance is not None and (
            not isinstance(opening_utterance, str) or not opening_utterance.strip()
        ):
            raise ValueError("opening_utterance must be nonempty text or None.")
        self.llm = llm
        self.system_prompt = system_prompt
        self.opening_prompt = opening_prompt
        self.opening_utterance = opening_utterance
        self.control = load_control(control_path)
        self.max_clarifications = max_clarifications
        self.max_rounds = max_rounds
        self.generated_turns = 0
        self.consecutive_clarifications = 0
        self.closed = False
        self._last_spoken = ""
        self.last_trace: dict = {}

    def respond(self, history: Sequence[DialogueMessage]) -> GenerationResult:
        if self.closed or (self.max_rounds is not None and self.generated_turns >= self.max_rounds):
            raise ValueError("This dialogue has ended; create a new therapist for another session.")
        turn = self.generated_turns + 1
        self.last_trace = {"generated_turn": turn, "model_input": [], "status": "pending"}
        # Earlier history is context, not a newly generated opening. Speaking
        # the supplied opening verbatim guarantees the intended first input.
        if turn == 1 and self.opening_utterance is not None:
            self.generated_turns = turn
            self._last_spoken = self.opening_utterance.strip()
            self.last_trace.update({
                "status": "completed", "source": "case_opening", "action": "continue",
                "consecutive_clarifications": 0, "stop_reason": None, "repeated_question": False,
            })
            return {
                "content": self.opening_utterance.strip(), "input_tokens": 0, "output_tokens": 0,
                "finish_reason": "eos", "dialogue_action": "continue", "stop_reason": None,
            }
        final_turn = self.max_rounds is not None and turn >= self.max_rounds
        instructions = [self.control["response_format"], self.control[
            "first_turn" if turn == 1 else "following_turn"
        ]]
        if self.consecutive_clarifications >= self.max_clarifications:
            instructions.append(self.control["repair_limit"])
        if final_turn:
            instructions.append(self.control["last_turn"])
        state = {
            "generated_turn": turn, "round_limit": self.max_rounds,
            "consecutive_clarifications": self.consecutive_clarifications,
            "max_clarifications": self.max_clarifications,
        }
        messages = [
            {"role": "system", "content": "\n\n".join(
                [self.system_prompt, *instructions, json.dumps(state)]
            )},
            # A transcript is input data, not plain-text assistant examples of
            # the new JSON output format. Never fabricate actions for old turns.
            {"role": "user", "content": json.dumps({
                "opening_instruction": self.opening_prompt,
                "conversation": [{"speaker": message.speaker, "content": message.content}
                                 for message in history],
            }, ensure_ascii=False)},
        ]
        self.last_trace["model_input"] = deepcopy(messages)
        self.last_trace["generation_attempts"] = []
        total_input_tokens = total_output_tokens = 0
        call_messages = messages
        try:
            # One normal generation plus, only when necessary, one format repair.
            # No public history or dialogue state advances until a valid reply.
            for attempt_number in range(2):
                attempt = {"kind": "generation" if attempt_number == 0 else "format_repair",
                           "model_input": deepcopy(call_messages), "status": "pending"}
                self.last_trace["generation_attempts"].append(attempt)
                try:
                    result = self.llm.reply(call_messages)
                except Exception as error:
                    attempt["status"] = "generation_failed"
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    raise
                attempt["raw_result"] = deepcopy(result)
                # Preserve the original draft for existing trace readers.
                if attempt_number == 0:
                    self.last_trace["raw_result"] = deepcopy(result)
                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                self.last_trace["usage"] = {
                    "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
                }
                try:
                    reply = _parse_reply(result["content"])
                except ValueError as error:
                    attempt["status"] = "invalid_format"
                    attempt["error"] = str(error)
                    if attempt_number == 1:
                        raise ValueError(
                            "Therapist JSON remained invalid after one format repair. "
                            "Inspect generation_attempts in therapist_trace.jsonl; "
                            "if finish_reason is length, increase max_new_tokens."
                        ) from error
                    call_messages = messages + [
                        {"role": "assistant", "content": result["content"]},
                        {"role": "user", "content": self.control["format_repair"]},
                    ]
                else:
                    attempt["status"] = "accepted"
                    self.last_trace["format_repaired"] = attempt_number == 1
                    break
        except Exception as error:
            self.last_trace.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        action = reply["action"]
        utterance = reply["utterance"]
        question = _question_key(utterance)
        repeated_question = question is not None and question == _question_key(self._last_spoken)
        if action == "continue" and repeated_question:
            # A literal repeated question is repair even if the model labels it
            # as progress. Acknowledgments and semantic rephrasings are excluded.
            action = "clarify"
        next_clarifications = self.consecutive_clarifications + 1 if action == "clarify" else 0
        # Closing may reflect completion, a wish to stop, or an unresolved item;
        # the control action alone does not prove that the activity succeeded.
        stop_reason = "therapist_close" if action == "close" else None
        override = None
        if next_clarifications > self.max_clarifications:
            action, utterance = "close", self.control["stall_closing"]
            stop_reason, override = "clarification_limit", "clarification_limit"
        elif final_turn:
            stop_reason = "round_limit"
            if action != "close":
                action, utterance = "close", self.control["budget_closing"]
                override = "round_limit"
        self.generated_turns = turn
        self.consecutive_clarifications = next_clarifications
        self.closed = action == "close"
        self._last_spoken = utterance
        self.last_trace.update({
            "status": "completed", "source": "model", "requested_action": reply["action"],
            "action": action, "consecutive_clarifications": next_clarifications,
            "stop_reason": stop_reason, "override": override, "spoken_utterance": utterance,
            "repeated_question": repeated_question,
        })
        return {**result, "content": utterance, "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens, "dialogue_action": action, "stop_reason": stop_reason}
