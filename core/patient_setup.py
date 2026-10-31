"""Prepare method resources before loading model weights; no data-source IO."""

import copy

from core.profiles import validate_patient_profile
from settings import PROJECT_DIR, read_prompt


def prepare_patient_setup(config: dict, method: str, patient_profile: dict) -> dict:
    validate_patient_profile(patient_profile)
    if method == "single_prompt":
        return {"system_prompt": read_prompt(config["patient_prompt"]), "options": {}, "prompts": {}}
    if method != "patient_agents":
        raise ValueError(f"Unknown patient method: {method}.")
    options = copy.deepcopy(config.get("patient_agents", {}))
    response_path = options.pop("response_prompt", str(PROJECT_DIR / "prompts" / "patient_agents" / "response.txt"))
    options.setdefault("policy_path", str(PROJECT_DIR / "patient_agents" / "policy.json"))
    response_prompt = read_prompt(response_path)
    from patient_agents.static import compile_profile

    compile_profile(patient_profile, options=options)
    return {"system_prompt": "", "options": options, "prompts": {"response": response_prompt}}
