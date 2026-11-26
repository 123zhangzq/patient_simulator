"""Evaluate saved patient sessions with the judge selected in the project config."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

# Allow invocation as either python evaluation/evaluate.py or python -m evaluation.evaluate.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DEFAULT_CONFIG, PROJECT_DIR, load_config
from core.providers import load_provider
from core.usage import print_usage, usage_report

# Questionnaire v1.0 item order: A1–A5 authenticity, B1–B5 training utility.
ITEMS = ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5")


def select_sessions(path):
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "dialogue.jsonl":
        path = path.parent
    if (path / "batch_metadata.json").is_file():
        batch = path
        candidates = sorted(p.parent for p in (batch / "sessions").glob("*/*/*/metadata.json"))
    elif (path / "metadata.json").is_file() and path.parent.parent.parent.name == "sessions":
        batch = path.parent.parent.parent.parent
        candidates = [path]
    else:
        raise ValueError("Use a run directory containing batch_metadata.json, a repeat directory, or its dialogue.jsonl.")
    selected = []
    for folder in candidates:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("status") == "completed":
            selected.append((folder, meta))
        else:
            print(f"Skipping {folder.relative_to(batch)} (status: {meta.get('status')}).", file=sys.stderr)
    if not selected:
        raise ValueError(f"No completed sessions to evaluate in {path}.")
    return batch, selected


def session_input(batch, folder, meta):
    """Read the two frozen prompts and only the public fields of the dialogue."""
    instance = folder.parent.name
    if meta["instance"] != instance:
        raise ValueError(f"Instance in metadata disagrees with folder {folder}.")
    payload = {}
    for role in ("patient", "therapist"):
        file = batch / "inputs" / instance / f"{role}.txt"
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta[f"{role}_sha256"]:
            raise ValueError(f"Input snapshot does not match the session metadata: {file}")
        payload[f"{role}_brief"] = data.decode("utf-8")
    turns = []
    raw = (folder / "dialogue.jsonl").read_bytes()
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        row = json.loads(line)
        turn = {k: row[k] for k in ("round", "speaker", "content", "source")}
        if (turn["speaker"] != ("Therapist", "Patient")[index % 2] or turn["round"] != index // 2 + 1
                or turn["source"] != ("fixed_opening" if index == 0 else "generated")
                or not isinstance(turn["content"], str) or not turn["content"].strip()):
            raise ValueError(f"Invalid dialogue record at line {index + 1} in {folder}.")
        turns.append(turn)
    if not turns or len(turns) != 2 * meta["completed_rounds"]:
        raise ValueError(f"Dialogue does not match completed_rounds in {folder}.")
    payload["dialogue"] = turns
    return payload, hashlib.sha256(raw).hexdigest()


def parse_rating(text, round_count):
    # Some models wrap valid JSON in a Markdown fence.
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    reply = json.loads(text)
    if not isinstance(reply, dict) or set(reply) != {"ratings"} or not isinstance(reply["ratings"], dict):
        raise ValueError("Expected a JSON object holding one ratings object.")
    ratings = reply["ratings"]
    if set(ratings) != set(ITEMS):
        raise ValueError("ratings must cover exactly the items A1–A5 and B1–B5.")
    for item in ITEMS:
        entry = ratings[item]
        if not isinstance(entry, dict) or set(entry) != {"score", "rationale", "evidence_rounds"}:
            raise ValueError(f"{item}: expected score, rationale, and evidence_rounds.")
        score, reason, rounds = entry["score"], entry["rationale"], entry["evidence_rounds"]
        if not (type(score) is int and 1 <= score <= 5 and isinstance(reason, str) and reason.strip()):
            raise ValueError(f"{item}: expected integer score 1–5 and a nonempty rationale.")
        if (not isinstance(rounds, list) or not rounds
                or any(type(r) is not int or not 1 <= r <= round_count for r in rounds)
                or len(rounds) != len(set(rounds))):
            raise ValueError(f"{item}: evidence_rounds must cite distinct rounds in this dialogue.")
    return {"ratings": {item: ratings[item] for item in ITEMS}}


def evaluate(config, batch, sessions, prompt, dry_run):
    alias = config["roles"].get("judge")
    if alias is None:
        raise ValueError("Set roles.judge to a model alias in config.json.")
    spec = config["models"][alias]
    generation = config["evaluation"]["generation"]
    print(f"Judge: {alias} ({spec['model_id']}); completed sessions selected: {len(sessions)}")
    if dry_run:
        for folder, meta in sessions:
            session_input(batch, folder, meta)
            print(f"  {folder.relative_to(batch)} ({meta['completed_rounds']} rounds)")
        return

    if spec["provider"] == "openrouter":
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env", override=False, interpolate=False)
    judge = load_provider(spec)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = batch / "evaluations" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"Evaluation output directory: {out}", flush=True)
    (out / "judge_prompt.txt").write_bytes(prompt.encode("utf-8"))
    settings = {"judge": alias, "provider": spec["provider"], "model_id": spec["model_id"],
                "generation": generation, "sessions": [s.relative_to(batch).as_posix() for s, _ in sessions]}
    (out / "metadata.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    failures = 0
    run_usage = []
    try:
        with (out / "results.jsonl").open("x", encoding="utf-8") as log:
            for folder, meta in sessions:
                session = folder.relative_to(batch).as_posix()
                payload, digest = session_input(batch, folder, meta)
                reply = judge.reply([{"role": "system", "content": prompt},
                                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                    generation=generation)
                result = {"session": session, "dialogue_sha256": digest,
                          "usage": {k: v for k, v in reply.items() if k != "content"}}
                run_usage.append({"speaker": "Judge", **result["usage"]})
                try:
                    rating = parse_rating(reply["content"], meta["completed_rounds"])
                except ValueError as exc:
                    result.update(status="invalid_rating", error=str(exc), raw_reply=reply["content"])
                    failures += 1
                else:
                    result.update(status="completed", **rating)
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                if result["status"] == "completed":
                    print(f"{session}: " + " ".join(f"{i}={result['ratings'][i]['score']}" for i in ITEMS))
                else:
                    print(f"{session}: {result['status']}")
    finally:
        settings["usage"] = usage_report(run_usage)
        (out / "metadata.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        print_usage(settings["usage"]["total"], "Evaluation total")
    if failures:
        raise ValueError(f"{failures} invalid judge response(s). Inspect {out / 'results.jsonl'}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Run directory, repeat directory, or dialogue.jsonl.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate and list sessions without calling the judge.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        batch, sessions = select_sessions(args.input)
        prompt = Path(__file__).with_name("judge_prompt.txt").read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError("judge_prompt.txt is empty.")
        evaluate(config, batch, sessions, prompt, args.dry_run)
    except (OSError, UnicodeError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
