"""Load the single public benchmark configuration. Credentials stay in the environment."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config.json"


def safe_id(value: object) -> bool:
    """Aliases and instance IDs are also used as portable directory names."""
    return (isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value) is not None
            and re.fullmatch(r"CON|PRN|AUX|NUL|(?:COM|LPT)[1-9]", value, re.I) is None)


def _keys(value, required, optional, name):
    if not isinstance(value, dict) or not required <= value.keys() <= required | optional:
        raise ValueError(f"Invalid fields in {name}. Required: {sorted(required)}. Optional: {sorted(optional)}.")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def validate_roles(roles, models):
    _keys(roles, {"therapist", "patients"}, {"judge"}, "roles")
    patients = roles["patients"]
    if not isinstance(patients, list) or not patients:
        raise ValueError("roles.patients must be a nonempty list of model aliases.")
    for alias in [roles["therapist"], *patients]:
        if not safe_id(alias) or alias not in models:
            raise ValueError(f"Select a model alias from config.models: {', '.join(models)}.")
    if len(set(patients)) != len(patients):
        raise ValueError("Select each patient model only once. Use run.repeats for repeated sessions.")
    if "judge" in roles and (not safe_id(roles["judge"]) or roles["judge"] not in models):
        raise ValueError("Set roles.judge to a model alias defined in config.models.")


def _validate_generation(gen, name):
    _keys(gen, {"max_new_tokens"}, {"temperature", "top_p", "top_k", "repetition_penalty", "seed"}, name)
    _positive_int(gen["max_new_tokens"], "max_new_tokens")
    if "top_k" in gen:
        _positive_int(gen["top_k"], "top_k")
    if "seed" in gen and (type(gen["seed"]) is not int or not 0 <= gen["seed"] < 2**32):
        raise ValueError("seed must be an integer from 0 to 2**32 - 1.")
    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in gen:
            x = gen[key]
            if type(x) not in (int, float) or not math.isfinite(x):
                raise ValueError(f"{key} must be a finite number.")
            if ((key == "temperature" and x < 0) or (key == "top_p" and not 0 < x <= 1)
                    or (key == "repetition_penalty" and x <= 0)):
                raise ValueError(f"{key} is outside its valid range.")


def _resolve(value, base, name):
    return str((base / Path(_text(value, name)).expanduser()).resolve())


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Resolve model/output paths relative to the configuration file (not the caller)."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    _keys(config, {"models", "roles", "run"}, {"evaluation"}, "config")
    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("models must contain at least one named model.")
    if len({name.casefold() for name in models}) != len(models):
        raise ValueError("Model aliases must remain distinct on case-insensitive filesystems.")
    for alias, model in models.items():
        if not safe_id(alias) or not isinstance(model, dict):
            raise ValueError("Each model needs a portable alphanumeric alias and a settings object.")
        provider = model.get("provider")
        if provider == "openrouter":
            _keys(model, {"provider", "model_id"}, {"api_key_env", "base_url", "timeout_seconds", "max_retries"}, alias)
            env = model.setdefault("api_key_env", "OPENROUTER_API_KEY")
            if not isinstance(env, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", env):
                raise ValueError(f"{alias}.api_key_env must name an environment variable.")
            url = urlsplit(_text(model.setdefault("base_url", "https://openrouter.ai/api/v1"), "base_url"))
            if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                    or url.password is not None or url.query or url.fragment):
                raise ValueError(f"{alias}.base_url must be an HTTP(S) endpoint without embedded credentials or query parameters.")
            if "timeout_seconds" in model:
                timeout = model["timeout_seconds"]
                if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError(f"{alias}.timeout_seconds must be a positive number.")
            if "max_retries" in model and (type(model["max_retries"]) is not int or model["max_retries"] < 0):
                raise ValueError(f"{alias}.max_retries must be a nonnegative integer.")
        elif provider == "local_qwen":
            _keys(model, {"provider", "model_id", "model_dir", "revision", "dtype", "max_context_tokens"}, set(), alias)
            model["model_dir"] = _resolve(model["model_dir"], path.parent, "model_dir")
            _text(model["revision"], "revision")
            if model["dtype"] not in ("auto", "float16", "bfloat16", "float32"):
                raise ValueError(f"Invalid dtype for {alias}.")
            _positive_int(model["max_context_tokens"], "max_context_tokens")
        else:
            raise ValueError(f"Choose openrouter or local_qwen as the provider for {alias}.")
        _text(model["model_id"], "model_id")
    validate_roles(config["roles"], models)
    run = config["run"]
    _keys(run, {"generation", "repeats", "output_dir"}, set(), "run")
    _validate_generation(run["generation"], "run.generation")
    _positive_int(run["repeats"], "run.repeats")
    run["output_dir"] = _resolve(run["output_dir"], path.parent, "output_dir")
    evaluation = config.setdefault("evaluation", {})
    _keys(evaluation, set(), {"generation"}, "evaluation")
    _validate_generation(evaluation.setdefault("generation", {"max_new_tokens": 4096}), "evaluation.generation")
    return config
