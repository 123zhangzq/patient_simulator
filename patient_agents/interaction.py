"""A bounded annotation of the last exchange, not an extra memory store."""

from copy import deepcopy


INTENTS = {"question", "instruction", "feedback", "closing", "other"}
ACTS = {"answer", "clarify", "choose", "plan", "report_attempt", "report_result", "pause", "close", "other"}
STATUSES = {"none", "proposed", "accepted", "attempt_reported", "completion_reported", "declined", "deferred"}
CLOSING_ACTS = {"close", "pause", "clarify", "other"}
CLOSING_STATUSES = {"none", "deferred", "declined"}


def visible_ids(context):
    return {item["id"] for key in ("accessible_current_units", "retrieved_memories")
            for item in context[key]}


def current_ids(context):
    return {item["id"] for item in context["accessible_current_units"]}


def response_schema(context):
    """Constrain syntax and available references; semantic checks still follow."""
    ids = sorted(current_ids(context))
    focus = {"type": "array", "minItems": 1 if ids else 0, "maxItems": 4 if ids else 0, "uniqueItems": True,
             "items": {"type": "string", **({"enum": ids} if ids else {})}}

    def branch(intents, acts, statuses):
        return {
            "type": "object", "additionalProperties": False,
            "required": ["therapist_intent", "patient_act", "action_status", "focus_ids"],
            "properties": {
                "therapist_intent": {"type": "string", "enum": sorted(intents)},
                "patient_act": {"type": "string", "enum": sorted(acts)},
                "action_status": {"type": "string", "enum": sorted(statuses)},
                "focus_ids": focus,
            },
        }

    return {
        "type": "object", "additionalProperties": False,
        "required": ["interaction", "utterance"],
        "properties": {
            "interaction": {"anyOf": [branch({"closing"}, CLOSING_ACTS, CLOSING_STATUSES),
                                      branch(INTENTS - {"closing"}, ACTS, STATUSES)]},
            "utterance": {"type": "string", "minLength": 1},
        },
    }


def validate_interaction(value, context):
    fields = {"therapist_intent", "patient_act", "action_status", "focus_ids"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("interaction requires therapist_intent, patient_act, action_status, focus_ids.")
    for key, choices in (("therapist_intent", INTENTS), ("patient_act", ACTS), ("action_status", STATUSES)):
        if not isinstance(value[key], str) or value[key] not in choices:
            raise ValueError(f"Invalid interaction.{key}; choose from {sorted(choices)}.")
    ids = value["focus_ids"]
    current = current_ids(context)
    if (not isinstance(ids, list) or len(ids) > 4
            or any(not isinstance(item, str) for item in ids)
            or (current and not ids) or len(set(ids)) != len(ids) or not set(ids) <= current):
        raise ValueError("focus_ids must contain 1–4 distinct current input IDs, not older memory IDs; empty only if no current input.")
    if value["therapist_intent"] == "closing":
        if value["patient_act"] not in CLOSING_ACTS or value["action_status"] not in CLOSING_STATUSES:
            raise ValueError("A closing requires patient_act close/pause/clarify/other and action_status none/deferred/declined.")
    return deepcopy(value)


class InteractionTracker:
    def __init__(self):
        self.previous = None

    def expose(self, context):
        """Forget the annotation when any of its supporting speech is inaccessible."""
        if self.previous is None:
            return None
        ids = self.previous["focus_ids"] + self.previous["patient_utterance_ids"]
        if not ids or not set(ids) <= visible_ids(context):
            self.previous = None
            return None
        return deepcopy(self.previous)

    def record(self, interaction, patient_utterance_ids):
        # Labels describe the generated report, never verified environment events.
        self.previous = ({**deepcopy(interaction),
                          "patient_utterance_ids": list(patient_utterance_ids)}
                         if interaction["focus_ids"] else None)
