"""Shared configuration for the Linux GPU download and dialogue entry points."""

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
PATH_FIELDS = (
    "model_dir", "therapist_prompt", "patient_prompt", "opening_prompt", "output_dir"
)
STRING_FIELDS = ("model_id", "revision", "dtype", *PATH_FIELDS)
POSITIVE_INT_FIELDS = ("rounds", "max_new_tokens", "max_context_tokens", "top_k")
EXPECTED_FIELDS = {
    *STRING_FIELDS, *POSITIVE_INT_FIELDS, "temperature", "top_p",
    "repetition_penalty", "seed",
}


def load_config(path: str | Path) -> tuple[dict, Path]:
    """Resolve configured paths relative to the JSON file, not the shell's cwd."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须是 JSON 对象。")
    missing = EXPECTED_FIELDS - config.keys()
    unknown = config.keys() - EXPECTED_FIELDS
    if missing or unknown:
        raise ValueError(f"配置字段错误：缺少 {sorted(missing)}，未知 {sorted(unknown)}。")
    for key in STRING_FIELDS:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} 必须是非空字符串。")
    for key in POSITIVE_INT_FIELDS:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须是正整数。")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed 必须是 0 到 2**32-1 之间的整数。")
    for key in ("temperature", "top_p", "repetition_penalty"):
        if type(config[key]) not in (int, float) or not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} 必须是有限的正数。")
    if config["top_p"] > 1:
        raise ValueError("top_p 不能大于 1。")
    if config["max_new_tokens"] >= config["max_context_tokens"]:
        raise ValueError("max_new_tokens 必须小于 max_context_tokens。")
    if config["dtype"] not in {"auto", "bfloat16", "float16"}:
        raise ValueError("dtype 只能是 auto、bfloat16 或 float16。")
    for key in PATH_FIELDS:
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = config_path.parent / value
        config[key] = str(value.resolve())
    return config, config_path


def read_prompt(path: str) -> str:
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt 文件为空：{path}")
    return content


def write_json(path: Path, data: dict) -> None:
    """Replace metadata atomically so an interrupted write leaves the prior file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
