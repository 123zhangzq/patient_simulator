"""Build a session-stable, evidence-linked person card before new dialogue.

Schema checks establish valid references, not clinical validity or semantic
grounding. The editable prompt and saved trace make those inferences reviewable.
"""

from collections.abc import Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from core.interfaces import DialogueMessage, LanguageModel
from patient_agents.structured import generate_object


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "patient_agents"
DIMENSIONS = {"behavior", "emotional_background", "communication", "goals_support"}
ITEM_FIELDS = {"dimension", "tendency", "when", "response_guidance", "evidence_ids", "confidence"}


def person_card_schema(sources):
    """Use only the supplied evidence IDs in a bounded, complete card."""
    ids = sorted(sources)
    return {
        "type": "object", "additionalProperties": False, "required": ["tendencies"],
        "properties": {
            "tendencies": {
                "type": "array", "maxItems": 3 if ids else 0,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["dimension", "tendency", "when", "response_guidance", "evidence_ids", "confidence"],
                    "properties": {
                        "dimension": {"type": "string", "enum": sorted(DIMENSIONS)},
                        **{field: {"type": "string", "minLength": 1, "maxLength": 180}
                           for field in ("tendency", "when", "response_guidance")},
                        "evidence_ids": {
                            "type": "array", "minItems": 1, "maxItems": 4, "uniqueItems": True,
                            "items": {"type": "string", **({"enum": ids} if ids else {})},
                        },
                        "confidence": {"type": "string", "enum": ["supported", "tentative"]},
                    },
                },
            },
        },
    }


class PersonCardBuilder:
    def __init__(self, llm: LanguageModel, compiled: dict, prompt: str | None = None,
                 repair_prompt: str | None = None, max_new_tokens: int = 768):
        self.llm = llm
        self.compiled = deepcopy(compiled)
        self.prompt = (PROMPT_DIR / "person_card.txt").read_text(encoding="utf-8").strip() if prompt is None else prompt
        self.repair_prompt = ((PROMPT_DIR / "format_repair.txt").read_text(encoding="utf-8").strip()
                              if repair_prompt is None else repair_prompt)
        for value in (self.prompt, self.repair_prompt):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Person-card prompts must be nonempty text.")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("Person-card max_new_tokens must be a positive integer.")
        self.max_new_tokens = max_new_tokens
        self.last_trace: dict = {}
        self._prefix_hash: str | None = None
        self._card: dict | None = None

    def _evidence(self, history: Sequence[DialogueMessage]) -> list[dict]:
        evidence = []
        view = self.compiled["response_profile"]
        for section in ("identity", "background"):
            for field, value in view[section].items():
                if value is not None and isinstance(value, (str, bool, int, float)):
                    evidence.append({"id": f"p{len(evidence) + 1}", "kind": "profile",
                                     "field": f"{section}.{field}", "value": value})
        for item in view["known_facts"]:
            evidence.append({"id": f"p{len(evidence) + 1}", "kind": "profile",
                             "field": "known_fact", "text": item["text"]})
        # Include measurements as dated evidence, not already-derived diagnoses
        # or personality labels. Internal policy thresholds are not evidence.
        for instrument, scores in self.compiled["evidence"].items():
            for field, value in scores.items():
                if type(value) in (int, float):
                    evidence.append({"id": f"p{len(evidence) + 1}", "kind": "assessment",
                                     "field": f"{instrument}.{field}", "value": value,
                                     "assessment_date": self.compiled.get("assessment_date")})
        evidence.extend({"id": f"h{index + 1}", "kind": "prior_utterance",
                         "speaker": message.speaker, "text": message.content}
                        for index, message in enumerate(history))
        return evidence

    @staticmethod
    def _validate(value: object, sources: dict[str, dict]) -> dict:
        if not isinstance(value, dict) or set(value) != {"tendencies"}:
            raise ValueError("Person card must contain only tendencies.")
        items = value["tendencies"]
        if not isinstance(items, list) or len(items) > 3:
            raise ValueError("Person card tendencies must be a list of at most three items.")
        normalized = []
        for item in items:
            if not isinstance(item, dict) or set(item) != ITEM_FIELDS:
                raise ValueError(f"Each tendency requires exactly {sorted(ITEM_FIELDS)}.")
            if not isinstance(item["dimension"], str) or item["dimension"] not in DIMENSIONS:
                raise ValueError("Unknown person-card dimension.")
            if item["confidence"] not in ("supported", "tentative"):
                raise ValueError("Tendency confidence must be supported or tentative.")
            for field in ("tendency", "when", "response_guidance"):
                if (not isinstance(item[field], str) or not item[field].strip()
                        or len(item[field].strip()) > 180):
                    raise ValueError(f"Tendency {field} must contain 1–180 characters.")
            ids = item["evidence_ids"]
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 4
                    or any(not isinstance(key, str) or key not in sources for key in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("Each tendency requires 1–4 distinct supplied evidence IDs.")
            if not any(sources[key]["kind"] != "prior_utterance"
                       or sources[key]["speaker"] == "Patient" for key in ids):
                raise ValueError("Therapist statements alone cannot establish a patient tendency.")
            normalized.append({**deepcopy(item), **{field: item[field].strip()
                              for field in ("tendency", "when", "response_guidance")}})
        return {"tendencies": normalized}

    def build(self, history: Sequence[DialogueMessage]) -> dict:
        prefix = tuple(history)
        if len(prefix) % 2 or any(
            not isinstance(message, DialogueMessage)
            or message.speaker != ("Therapist" if index % 2 == 0 else "Patient")
            or not isinstance(message.content, str) or not message.content.strip()
            for index, message in enumerate(prefix)
        ):
            raise ValueError("Person-card history must contain complete, nonempty Therapist/Patient pairs.")
        serialized = json.dumps([{"speaker": m.speaker, "content": m.content} for m in prefix],
                                ensure_ascii=False, sort_keys=True)
        prefix_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if self._prefix_hash is not None:
            if prefix_hash != self._prefix_hash:
                raise ValueError("A session person card cannot be rebuilt from later or different history.")
            if self._card is None:
                raise ValueError("Person-card initialization failed; start a fresh patient instance.")
            return deepcopy(self._card)
        self._prefix_hash = prefix_hash
        evidence = self._evidence(prefix)
        sources = {item["id"]: item for item in evidence}
        payload = {"task": "build_person_card", "evidence": evidence,
                   "unknowns": deepcopy(self.compiled["unknowns"]),
                   "simulation_date": self.compiled.get("simulation_date")}
        messages = [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        self.last_trace = {"status": "building", "prefix_sha256": prefix_hash,
                           "model_input": deepcopy(messages), "evidence": deepcopy(evidence),
                           "max_new_tokens": self.max_new_tokens}
        try:
            card, result = generate_object(
                self.llm, messages, lambda value: self._validate(value, sources),
                self.repair_prompt, self.last_trace, max_new_tokens=self.max_new_tokens,
                json_schema=person_card_schema(sources),
            )
        except Exception:
            self.last_trace["status"] = "failed"
            raise
        self._card = deepcopy(card)
        self.last_trace.update(status="ready", card=deepcopy(card), generation_result=deepcopy(result))
        return deepcopy(card)
