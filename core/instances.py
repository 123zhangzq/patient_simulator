"""Read the released benchmark dataset using only its manifest and role texts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.config import safe_id


def load_instances(path: str | Path, *, allow_edited_prompts: bool = False) -> list[dict]:
    root = Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("version"), str) or not manifest["version"].strip():
        raise ValueError("manifest.json requires a dataset version string.")
    entries = manifest.get("instances")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.json requires a nonempty instances list.")
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not safe_id(entry.get("id")):
            raise ValueError("Invalid instance ID in manifest.json.")
        instance_id = entry["id"]
        if instance_id.casefold() in seen:
            raise ValueError(f"Duplicate instance ID: {instance_id}")
        seen.add(instance_id.casefold())
        if (entry.get("mode") not in ("continuation", "new_activity") or not safe_id(entry.get("category"))
                or type(entry.get("max_rounds")) is not int or entry["max_rounds"] < 1):
            raise ValueError(f"Invalid category, mode or max_rounds for {instance_id}.")
        instance = dict(entry, dataset_version=manifest["version"], modified=False)
        for role in ("therapist", "patient"):
            file = root / instance_id / f"{role}.txt"
            data = file.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            field = role + "_sha256"
            expected = entry.get(field)
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError(f"Invalid {field} for {instance_id} in manifest.json.")
            if digest != expected:
                if not allow_edited_prompts:
                    raise ValueError(f"{file} differs from the dataset manifest. To run an edited copy, use --allow-edited-prompts.")
                instance["modified"] = True
            prompt = data.decode("utf-8")
            if not prompt.strip():
                raise ValueError(f"Empty prompt: {file}")
            instance[field] = digest
            instance[role] = prompt
        lines = instance["therapist"].splitlines()
        marker = "[Opening utterance]"
        if lines.count(marker) != 1:
            raise ValueError(f"{instance_id}/therapist.txt requires one {marker} section at the end.")
        opening = "\n".join(lines[lines.index(marker) + 1:]).strip()
        if not opening or "\n\n" in opening:
            raise ValueError(f"Expected a final opening paragraph in {instance_id}/therapist.txt.")
        instance["opening"] = opening
        result.append(instance)
    return result
