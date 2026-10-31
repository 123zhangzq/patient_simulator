"""Case-contained patient information shared by every simulation method.

Source workbooks are deliberately not a runtime dependency. Provenance is kept
for auditing but is excluded from model-facing patient information.
"""

import copy
from datetime import date
import hashlib
import json
import math


PROFILE_FIELDS = {
    "schema_version", "participant_id", "role", "assessment_date", "simulation_date",
    "background", "assessments", "known_facts", "initial_memories", "provenance", "unknowns",
}
MOCA_LIMITS = {
    "total": 30, "attention": 6, "language": 3, "delayed_recall": 5,
    "category_cue": 5, "multiple_choice_cue": 5, "visuospatial_executive": 5,
    "abstraction": 2, "orientation": 6, "naming": 3,
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _number(value, name, maximum=None):
    if value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number or null.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}; retain the source value in provenance and use null pending review.")


def validate_patient_profile(profile: object) -> None:
    if not isinstance(profile, dict):
        raise ValueError("patient_profile must be a JSON object embedded in case.json.")
    missing = PROFILE_FIELDS - profile.keys()
    unknown = profile.keys() - PROFILE_FIELDS
    if missing or unknown:
        raise ValueError(f"Invalid patient_profile fields: missing {sorted(missing)}, unknown {sorted(unknown)}.")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("patient_profile.schema_version must be 1.")
    _text(profile["participant_id"], "patient_profile.participant_id")
    if profile["role"] != "stroke_survivor":
        raise ValueError("patient_profile.role must be stroke_survivor; caregiver data must not be used as patient data.")
    dates = {}
    for key in ("assessment_date", "simulation_date"):
        value = profile[key]
        if value is None and key == "assessment_date":
            continue
        try:
            if not isinstance(value, str) or len(value) != 10:
                raise ValueError
            dates[key] = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"patient_profile.{key} must be an ISO YYYY-MM-DD date.") from None
    if dates.get("assessment_date", dates["simulation_date"]) > dates["simulation_date"]:
        raise ValueError("patient_profile.assessment_date must not be after simulation_date.")
    background = profile["background"]
    if not isinstance(background, dict) or not {"age", "language", "education_years"} <= background.keys():
        raise ValueError("patient_profile.background requires age, language and education_years (null if unknown).")
    for key, value in background.items():
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError(f"patient_profile.background.{key} must be a scalar or null.")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"patient_profile.background.{key} must be finite.")
    _number(background["age"], "patient_profile.background.age", 120)
    _number(background["education_years"], "patient_profile.background.education_years", 60)
    if background["language"] is not None:
        _text(background["language"], "patient_profile.background.language")
    assessments = profile["assessments"]
    if not isinstance(assessments, dict) or set(assessments) != {"moca", "sdmt", "dass21"}:
        raise ValueError("patient_profile.assessments requires moca, sdmt and dass21 objects.")
    for group, required in (("moca", set(MOCA_LIMITS)), ("sdmt", {"total"}),
                            ("dass21", {"depression", "anxiety", "stress"})):
        values = assessments[group]
        if not isinstance(values, dict) or not required <= values.keys():
            raise ValueError(f"patient_profile.assessments.{group} requires {sorted(required)} (null if unknown).")
        for key, value in values.items():
            limit = MOCA_LIMITS.get(key) if group == "moca" else (42 if group == "dass21" else None)
            _number(value, f"patient_profile.assessments.{group}.{key}", limit)
    for group in ("known_facts", "initial_memories"):
        if not isinstance(profile[group], list):
            raise ValueError(f"patient_profile.{group} must be a list.")
        ids = set()
        for item in profile[group]:
            if not isinstance(item, dict) or not {"id", "text", "source"} <= item.keys() or item.keys() - {"id", "text", "source", "cue_tags"}:
                raise ValueError(f"patient_profile.{group} entries require id, text and source; cue_tags is optional.")
            for key in ("id", "text", "source"):
                _text(item[key], f"patient_profile.{group}.{key}")
            if item["id"] in ids:
                raise ValueError(f"Duplicate patient_profile.{group} id: {item['id']}.")
            ids.add(item["id"])
            if "cue_tags" in item:
                if not isinstance(item["cue_tags"], list):
                    raise ValueError(f"patient_profile.{group}.cue_tags must be a list.")
                for tag in item["cue_tags"]:
                    _text(tag, f"patient_profile.{group}.cue_tags item")
    if not isinstance(profile["provenance"], dict):
        raise ValueError("patient_profile.provenance must be an object.")
    if not isinstance(profile["unknowns"], list):
        raise ValueError("patient_profile.unknowns must be a list.")
    for value in profile["unknowns"]:
        _text(value, "patient_profile.unknowns item")
    # Reject non-JSON values/NaN even inside audit-only metadata.
    try:
        json.dumps(profile, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"patient_profile must contain finite JSON data: {error}") from error


def patient_information(profile: dict) -> dict:
    """Identical source information available to baseline and proposed methods."""
    validate_patient_profile(profile)
    result = copy.deepcopy({key: value for key, value in profile.items()
                            if key not in {"provenance", "assessment_date", "simulation_date"}})
    for group in ("known_facts", "initial_memories"):
        result[group] = [{key: value for key, value in item.items() if key != "source"}
                         for item in result[group]]
    return result


def render_patient_profile(profile: dict) -> str:
    return json.dumps(patient_information(profile), ensure_ascii=False, indent=2, allow_nan=False)


def profile_fingerprint(profile: dict) -> str:
    validate_patient_profile(profile)
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
