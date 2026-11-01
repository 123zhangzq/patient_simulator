"""Prepare method resources before loading model weights; no data-source IO."""

import copy

from core.profiles import validate_patient_profile
from settings import PROJECT_DIR, read_prompt


def prepare_patient_setup(config: dict, method: str, patient_profile: dict, shared_context: str = "") -> dict:
    validate_patient_profile(patient_profile)
    if not isinstance(shared_context, str):
        raise ValueError("shared_context must be text.")
    situation = "Shared situation\n" + shared_context.strip() if shared_context.strip() else ""
    if method == "single_prompt":
        prompt = "\n\n".join(part for part in (read_prompt(config["patient_prompt"]), situation) if part)
        return {"system_prompt": prompt, "options": {}, "prompts": {}}
    if method != "patient_agents":
        raise ValueError(f"Unknown patient method: {method}.")
    options = copy.deepcopy(config.get("patient_agents", {}))
    response_path = options.pop("response_prompt", str(PROJECT_DIR / "prompts" / "patient_agents" / "response.txt"))
    options.setdefault("policy_path", str(PROJECT_DIR / "patient_agents" / "policy.json"))
    response_prompt = read_prompt(response_path)
    from patient_agents.static import compile_profile

    compile_profile(patient_profile, options=options)
    return {"system_prompt": situation, "options": options, "prompts": {"response": response_prompt}}
