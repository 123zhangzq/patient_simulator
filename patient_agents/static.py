"""Compile measured evidence into explicit, replaceable simulation assumptions.

The numerical policy is an experimental mechanism, not a validated conversion
from a screening score to a person's cognitive capacity or diagnosis.
"""

from copy import deepcopy
import json
import math
from pathlib import Path


POLICY_PATH = Path(__file__).with_name("policy.json")
TRAIT_VALUES = {
    "attention": {"preserved", "reduced", "markedly_reduced", "unknown"},
    "memory": {"preserved", "reduced", "markedly_reduced", "unknown"},
    "language": {"preserved", "reduced", "unknown"},
}
OPTION_FIELDS = {"attention_enabled", "memory_enabled", "policy_path", "parameters", "traits"}
BACKGROUND_FACT_CUES = {
    "education_years": ["education", "school", "study", "studied", "education_years", "学习", "上学"],
    "living": ["live", "living", "home", "family", "household", "住", "家里"],
    "lives_alone": ["alone", "live", "living", "home", "住", "独居"],
    "lives_with_family": ["family", "live", "living", "home", "住", "家里"],
    "lives_with_friends": ["friends", "live", "living", "home", "住", "朋友"],
    "lives_with_helper": ["helper", "caregiver", "live", "living", "home", "住", "照护"],
    "lives_with_others": ["live", "living", "home", "household", "住", "家里"],
    "independence": ["independent", "independently", "independence", "assistance", "daily", "help", "自理"],
    "stroke_year": ["stroke", "stroke_year", "中风"],
}


def load_policy(options: dict | None = None) -> dict:
    options = {} if options is None else options
    if not isinstance(options, dict) or options.keys() - OPTION_FIELDS:
        raise ValueError(f"Unknown patient_agents options; allowed: {sorted(OPTION_FIELDS)}")
    for name in ("attention_enabled", "memory_enabled"):
        if name in options and type(options[name]) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    if "policy_path" in options and not isinstance(options["policy_path"], (str, Path)):
        raise ValueError("policy_path must be a filesystem path.")
    defaults = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    path = Path(options.get("policy_path", POLICY_PATH))
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.keys() != defaults.keys():
        raise ValueError("Policy must contain version, status, parameters and marker lists.")
    if not isinstance(policy["version"], str) or not policy["version"].strip():
        raise ValueError("Policy version must be a nonempty string.")
    parameters = policy["parameters"]
    overrides = options.get("parameters", {})
    if (not isinstance(parameters, dict) or parameters.keys() != defaults["parameters"].keys()
            or not isinstance(overrides, dict) or overrides.keys() - parameters.keys()):
        raise ValueError("Policy parameter keys must match the documented default policy.")
    parameters.update(overrides)
    for name, value in parameters.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Policy parameter {name} must be a finite number.")
        if "probability" in name or "retention_per_turn" in name or name == "cue_bonus":
            if not 0 <= value <= 1:
                raise ValueError(f"Policy parameter {name} must lie between 0 and 1.")
        elif type(value) is not int or value < 1:
            raise ValueError(f"Policy parameter {name} must be a positive integer.")
    for trait, maximum in (("attention", 6), ("memory", 5)):
        lower = parameters[f"{trait}_reduced_min"]
        upper = parameters[f"{trait}_preserved_min"]
        if not 1 <= lower < upper <= maximum:
            raise ValueError(
                f"{trait} thresholds must satisfy 1 <= reduced_min < preserved_min <= {maximum}."
            )
    if parameters["language_preserved_min"] > 3:
        raise ValueError("language_preserved_min must not exceed the MoCA language maximum of 3.")
    for name in ("cue_markers", "recall_markers", "stopwords"):
        if not isinstance(policy[name], list) or not all(
            isinstance(item, str) and item.strip() for item in policy[name]
        ):
            raise ValueError(f"{name} must be a list of nonempty strings.")
    traits = options.get("traits", {})
    if not isinstance(traits, dict) or traits.keys() - TRAIT_VALUES.keys():
        raise ValueError("traits supports only attention, memory, and language.")
    for name, value in traits.items():
        if not isinstance(value, str) or value not in TRAIT_VALUES[name]:
            raise ValueError(f"Invalid {name} trait: {value!r}.")
    return policy


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def compile_profile(profile: dict, options: dict | None = None) -> dict:
    """Return auditable settings plus a deliberately restricted response view."""
    from core.profiles import profile_fingerprint, validate_patient_profile

    validate_patient_profile(profile)
    options = {} if options is None else options
    policy = load_policy(options)
    parameters = policy["parameters"]
    assessments = profile.get("assessments", {})
    moca = assessments.get("moca", {})
    traits, derivation = {}, {}
    for trait, field in (("attention", "attention"), ("memory", "delayed_recall")):
        value = _number(moca.get(field))
        band = "unknown" if value is None else (
            "preserved" if value >= parameters[f"{trait}_preserved_min"] else
            "reduced" if value >= parameters[f"{trait}_reduced_min"] else "markedly_reduced"
        )
        traits[trait] = band
        derivation[trait] = {"field": f"assessments.moca.{field}", "value": value,
                             "rule": f"{trait}_score_band", "assumption": band}
    value = _number(moca.get("language"))
    traits["language"] = "unknown" if value is None else (
        "preserved" if value >= parameters["language_preserved_min"] else "reduced"
    )
    derivation["language"] = {"field": "assessments.moca.language", "value": value,
                              "rule": "language_score_band", "assumption": traits["language"]}
    for trait, value in options.get("traits", {}).items():
        traits[trait] = value
        derivation[trait]["override"] = value
    background = profile.get("background", {})
    identity = {"role": profile["role"]}
    for field in ("age", "language"):
        if field in background and background[field] is not None:
            identity[field] = background[field]
    # These are available personal facts, not test scores or retrieved episodes.
    facts = deepcopy(profile.get("known_facts", []))
    for field, tags in BACKGROUND_FACT_CUES.items():
        value = background.get(field)
        if value is not None:
            facts.append({"id": f"background:{field}", "text": f"{field}: {value}",
                          "source": f"background.{field}", "cue_tags": tags})
    return {
        "participant_id": profile["participant_id"],
        "profile_fingerprint": profile_fingerprint(profile),
        "assessment_date": profile.get("assessment_date"),
        "simulation_date": profile.get("simulation_date"),
        "policy": policy,
        "traits": traits,
        "derivation": derivation,
        "evidence": deepcopy(assessments),
        "unknowns": deepcopy(profile.get("unknowns", [])),
        "response_profile": {"identity": identity, "expression": {"language": traits["language"]}},
        "known_facts": facts,
        "initial_memories": deepcopy(profile.get("initial_memories", [])),
        "limits": [
            "score_to_mechanism_mapping_is_an_unvalidated_heuristic",
            "moca_total_does_not_set_memory",
            "sdmt_and_dass_are_retained_as_evidence_without_unvalidated_cutoffs",
            "language_screening_does_not_establish_aphasia_or_a_personality",
            "unknown_traits_use_explicit_default_parameters",
        ],
    }
