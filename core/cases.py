"""Load self-contained cases; keep therapist and patient inputs separate."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

from core.profiles import validate_patient_profile


_IDENTIFIER = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_REQUIRED_FIELDS = {"schema_version", "id", "category", "title", "summary", "therapist", "patient_profile"}
_OPTIONAL_FIELDS = {"source", "rounds"}
_THERAPIST_FIELDS = {
    "context", "objectives", "materials", "opening_question", "conversation_plan",
    "response_guidance", "completion_criteria",
}
_OPTIONAL_THERAPIST_FIELDS = {"reference_utterances"}
_LIST_FIELDS = {
    "objectives", "materials", "conversation_plan", "response_guidance", "completion_criteria",
}


@dataclass(frozen=True)
class DialogueCase:
    path: Path
    data: dict

    @property
    def case_id(self) -> str:
        return self.data["id"]


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _range(value: object, name: str) -> None:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(item) is not int or item < 1 for item in value)
            or value[0] > value[1]):
        raise ValueError(f"{name} must be [first, last] with positive, ordered integers.")


def _validate(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("A case file must contain a JSON object.")
    missing = _REQUIRED_FIELDS - data.keys()
    unknown = data.keys() - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing or unknown:
        raise ValueError(f"Invalid case fields: missing {sorted(missing)}, unknown {sorted(unknown)}.")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1.")
    validate_patient_profile(data["patient_profile"])
    for key in ("id", "category", "title", "summary"):
        _text(data[key], key)
    for key in ("id", "category"):
        if len(data[key]) > 80 or not _IDENTIFIER.fullmatch(data[key]):
            raise ValueError(f"{key} must use lowercase letters, digits and single underscores (max 80 characters).")
    if "rounds" in data and (type(data["rounds"]) is not int or data["rounds"] < 1):
        raise ValueError("rounds must be a positive integer.")
    therapist = data["therapist"]
    if (not isinstance(therapist, dict) or not _THERAPIST_FIELDS <= therapist.keys()
            or therapist.keys() - _THERAPIST_FIELDS - _OPTIONAL_THERAPIST_FIELDS):
        raise ValueError(
            f"therapist requires {', '.join(sorted(_THERAPIST_FIELDS))}; "
            "reference_utterances is optional."
        )
    for key in ("context", "opening_question"):
        _text(therapist[key], f"therapist.{key}")
    for key in _LIST_FIELDS:
        values = therapist[key]
        if not isinstance(values, list) or (not values and key != "materials"):
            raise ValueError(f"therapist.{key} must be a list of nonempty strings.")
        for value in values:
            _text(value, f"therapist.{key} item")
    if "reference_utterances" in therapist:
        examples = therapist["reference_utterances"]
        if not isinstance(examples, list):
            raise ValueError("therapist.reference_utterances must be a list of nonempty strings.")
        for example in examples:
            _text(example, "therapist.reference_utterances item")
    if "source" in data:
        source = data["source"]
        if not isinstance(source, dict):
            raise ValueError("source must be a JSON object when provided.")
        for key in ("workbook", "worksheet"):
            _text(source.get(key), f"source.{key}")
        for key in ("excel_rows", "utt_ids"):
            _range(source.get(key), f"source.{key}")
        dialogs = source.get("dialog_ids")
        if not isinstance(dialogs, list):
            raise ValueError("source.dialog_ids must be a list of strings.")
        for value in dialogs:
            _text(value, "source.dialog_ids item")
        if "materials_excel_rows" in source:
            _range(source["materials_excel_rows"], "source.materials_excel_rows")


def load_cases(path: str | Path) -> tuple[DialogueCase, ...]:
    """Find case.json recursively; reference transcripts are never loaded here."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.rglob("case.json"), key=lambda item: item.relative_to(root).as_posix())
    else:
        raise FileNotFoundError(f"The cases path does not exist: {root}")
    if not paths:
        raise ValueError(f"No case.json files found in {root}.")
    cases = []
    known_ids: dict[str, Path] = {}
    for case_path in paths:
        try:
            data = json.loads(case_path.read_text(encoding="utf-8"))
            _validate(data)
        except (ValueError, OSError) as error:
            raise ValueError(f"Invalid case file {case_path}: {error}") from error
        case_id = data["id"]
        if case_id in known_ids:
            raise ValueError(f"Duplicate case id {case_id!r}: {known_ids[case_id]} and {case_path}.")
        known_ids[case_id] = case_path
        cases.append(DialogueCase(path=case_path, data=data))
    return tuple(cases)


def select_cases(
    cases: Sequence[DialogueCase],
    case_ids: Sequence[str] | None = None,
    category: str | None = None,
) -> tuple[DialogueCase, ...]:
    """Apply optional filters without changing discovery order."""
    if case_ids:
        missing = set(case_ids) - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}.")
    if category is not None and category not in {case.data["category"] for case in cases}:
        raise ValueError(f"Unknown case category: {category!r}.")
    selected = tuple(
        case for case in cases
        if (not case_ids or case.case_id in case_ids)
        and (category is None or case.data["category"] == category)
    )
    if not selected:
        raise ValueError("No cases match the requested case ids and category.")
    return selected


def compose_prompts(
    case: DialogueCase, therapist_template: str, opening_template: str,
) -> dict[str, str]:
    """Use an explicit field allowlist so provenance and patient speech stay out."""
    _validate(case.data)
    _text(therapist_template, "therapist template")
    _text(opening_template, "opening template")
    fields = []
    for _, field, format_spec, conversion in Formatter().parse(opening_template):
        if field is not None:
            if field != "opening_question" or format_spec or conversion:
                raise ValueError("The opening template only supports {opening_question}, without formatting or conversion.")
            fields.append(field)
    if fields != ["opening_question"]:
        raise ValueError("The opening template must contain {opening_question} exactly once.")
    therapist = case.data["therapist"]
    sections = [therapist_template.strip(), "Conversation context\n" + therapist["context"].strip()]
    for key, title in (
        ("objectives", "Conversation objectives"),
        ("materials", "Task material and reference information"),
        ("conversation_plan", "Suggested conversation progression"),
        ("response_guidance", "How to respond to this patient as the dialogue unfolds"),
        ("completion_criteria", "What a useful conclusion would establish"),
    ):
        if therapist[key]:
            sections.append(title + "\n" + "\n".join(f"- {item}" for item in therapist[key]))
    if therapist.get("reference_utterances"):
        sections.append(
            "Historical therapist wording examples\n"
            "These selected source excerpts illustrate questions and responses for this activity. "
            "They are examples, not messages already spoken in this simulation or a script to replay. "
            "Use a line only when it fits this patient's actual response; adapt person references as needed.\n"
            + "\n".join(f"- {item}" for item in therapist["reference_utterances"])
        )
    return {
        "therapist": "\n\n".join(sections),
        "opening": opening_template.format(opening_question=therapist["opening_question"]).strip(),
    }
