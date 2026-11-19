"""Run independent single-prompt patient sessions and save their public dialogue."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.providers import load_provider


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _messages(prompt, history, speaker):
    # A labeled transcript works for either participant, including the therapist
    # whose first line was fixed. Only newly spoken turns are appended here.
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({
            "conversation": history, "next_speaker": speaker,
        }, ensure_ascii=False)},
    ]


def _generate(model, prompt, history, speaker, generation, number):
    parameters = dict(generation)
    if "seed" in parameters:
        # Equivalent roles and turns get the same requested seed across model comparisons.
        parameters["seed"] = (parameters["seed"] + 2 * (number - 1) + int(speaker == "Patient")) % 2**32
    reply = model.reply(_messages(prompt, history, speaker), generation=parameters)
    content = reply["content"].strip()
    closing = speaker == "Therapist" and content.splitlines()[-1:] == ["[END_SESSION]"]
    if closing:
        content = content[: -len("[END_SESSION]")].strip()
    if not content:
        raise ValueError(f"{speaker} returned no spoken text in round {number}.")
    usage = {k: reply[k] for k in ("input_tokens", "output_tokens", "finish_reason",
                                    "model", "revision", "request_id") if k in reply}
    if "seed" in parameters:
        usage["requested_seed"] = parameters["seed"]
    return content, closing, usage


def _session(folder, instance, therapist, patient, generation, repeat, info):
    folder.mkdir(parents=True, exist_ok=False)
    history = []
    early_history = instance["therapist"].partition("\n[Earlier conversation]\n")[2].split("\n\n", 1)[0].strip()
    info.update(status="running", repeat=repeat, completed_rounds=0, max_rounds=instance["max_rounds"],
                generation=generation, therapist_sha256=instance["therapist_sha256"],
                patient_sha256=instance["patient_sha256"], dataset_version=instance["dataset_version"],
                mode=instance["mode"], modified=instance.get("modified", False))
    meta_path = folder / "metadata.json"
    _write_json(meta_path, info)
    try:
        with (folder / "dialogue.jsonl").open("w", encoding="utf-8") as log, (
            folder / "dialogue.md"
        ).open("w", encoding="utf-8") as readable:
            readable.write(f"# {instance['id']}\n\n")
            if early_history:
                readable.write("## Supplied earlier conversation\n\n")
                readable.write(early_history.replace("\n", "\n\n") + "\n\n")
            readable.write("## New dialogue\n\n")
            for number in range(1, instance["max_rounds"] + 1):
                closing = False
                for speaker, model in (("Therapist", therapist), ("Patient", patient)):
                    supplied = speaker == "Therapist" and number == 1
                    usage = {}
                    if supplied:
                        content = instance["opening"]
                    else:
                        content, wants_close, usage = _generate(
                            model, instance[speaker.lower()], history, speaker, generation, number)
                        closing = closing or wants_close
                    turn = {"speaker": speaker, "content": content}
                    history.append(turn)
                    source = "fixed_opening" if supplied else "generated"
                    record = {"round": number, **turn, "source": source, **usage}
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    label = f"Round {number} · {speaker}" + (" (fixed opening)" if supplied else "")
                    readable.write(f"**{label}**\n\n{content}\n\n")
                    readable.flush()
                    print(f"[Round {number}] {speaker}: {content}", flush=True)
                info["completed_rounds"] = number
                if closing:
                    break
            info.update(status="completed", stop_reason="therapist_close" if closing else "round_limit")
    except (Exception, KeyboardInterrupt) as exc:
        info["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        info["error_type"] = type(exc).__name__
        raise
    finally:
        _write_json(meta_path, info)


def run(config, instances):
    """Benchmark every selected patient against one fixed therapist."""
    models, roles, options = config["models"], config["roles"], config["run"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(options["output_dir"]) / stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output}", flush=True)
    # Only record public model settings, never credentials or request headers.
    public_fields = ("provider", "model_id", "revision", "dtype", "max_context_tokens")
    meta = {"status": "running", "roles": roles, "models": {
        alias: {k: models[alias][k] for k in public_fields if k in models[alias]}
        for alias in (roles["therapist"], *roles["patients"])},
        "instances": [item["id"] for item in instances], "repeats": options["repeats"],
        "generation": options["generation"], "completed_sessions": 0}
    batch_path = output / "batch_metadata.json"
    _write_json(batch_path, meta)
    try:
        for instance in instances:
            directory = output / "inputs" / instance["id"]
            directory.mkdir(parents=True)
            for role in ("therapist", "patient"):
                (directory / f"{role}.txt").write_bytes(instance[role].encode("utf-8"))
        therapist_spec = models[roles["therapist"]]
        therapist = load_provider(therapist_spec)
        for alias in roles["patients"]:
            patient = therapist if models[alias] == therapist_spec else load_provider(models[alias])
            for instance in instances:
                for repeat in range(1, options["repeats"] + 1):
                    generation = dict(options["generation"])
                    if generation.get("seed") is not None:
                        generation["seed"] = (generation["seed"] + repeat - 1) % 2**32
                    folder = output / "sessions" / alias / instance["id"] / f"repeat_{repeat:03d}"
                    print(f"\n{alias} / {instance['id']} / repeat {repeat}", flush=True)
                    _session(folder, instance, therapist, patient, generation, repeat,
                             {"instance": instance["id"], "therapist": roles["therapist"], "patient": alias})
                    meta["completed_sessions"] += 1
                    _write_json(batch_path, meta)
            del patient
        meta["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        meta["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["error_type"] = type(exc).__name__
        raise
    finally:
        _write_json(batch_path, meta)
    return output
