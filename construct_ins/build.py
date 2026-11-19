"""Build the published two-text instances from reviewed authoring files (stdlib only)."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "instances"
VERSION = "1.0"
BACKGROUND = (
    "age", "gender", "language", "education_years", "lives_with_family",
    "lives_with_helper", "independence", "stroke_year",
)
THERAPIST_SECTIONS = (
    ("objectives", "Activity objectives"),
    ("materials", "Available materials and reference information"),
    ("conversation_plan", "Suggested progression"),
    ("response_guidance", "Guidance for this activity"),
    ("completion_criteria", "When to finish"),
    ("reference_utterances", "Optional examples of therapist wording"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(data):
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def one_line(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip().splitlines()) != 1:
        raise ValueError("Expected a single nonempty line of dialogue.")
    return value.strip()


def history_section(history, mode):
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError("Earlier conversation must contain complete therapist/patient exchanges.")
    if mode != ("continuation" if history else "new_activity"):
        raise ValueError("Only continuations may contain earlier exchanges, and must provide them.")
    lines = []
    for index, turn in enumerate(history):
        speaker = ("Therapist", "Patient")[index % 2]
        if turn["speaker"] != speaker:
            raise ValueError(f"Expected {speaker} at history entry {index + 1}.")
        lines.append(f"{speaker}: {one_line(turn['content'])}")
    return "[Earlier conversation]\n" + "\n".join(lines) if lines else ""


def patient_information(profile):
    """Select role material explicitly; the adjacent provenance stays in the source case."""
    measures = {group: profile["assessments"][group] for group in ("moca", "sdmt", "dass21")}
    for fields in measures.values():
        if not isinstance(fields, dict) or any(
            value is not None and type(value) not in (int, float) for value in fields.values()
        ):
            raise ValueError("Assessment entries must be numeric or null.")
    result = {
        "role": profile["role"],
        "background": {k: profile["background"][k] for k in BACKGROUND if k in profile["background"]},
        "assessments": measures,
    }
    for key in ("known_facts", "initial_memories"):
        if profile[key]:
            result[key] = [entry["text"] for entry in profile[key]]
    return json_text(result)


def render(case, scenario, templates):
    therapist = scenario["therapist"]
    shared = "[Shared situation]\n" + scenario["shared_context"]
    earlier = history_section(case["session"]["initial_history"], scenario["mode"])
    therapist_parts = [templates["therapist"], shared, "[Activity context]\n" + therapist["context"]]
    for key, heading in THERAPIST_SECTIONS:
        if therapist.get(key):
            therapist_parts.append(f"[{heading}]\n" + "\n".join(f"- {line}" for line in therapist[key]))
    therapist_parts.extend([earlier, "[Opening utterance]\n" + one_line(therapist["opening_question"])])
    patient_parts = [
        templates["patient"], shared,
        "[Patient information]\n" + patient_information(case["patient_profile"]),
    ]
    notes = case.get("patient_role_notes", [])
    if notes:
        patient_parts.append("[Role notes]\n" + "\n".join(f"- {note}" for note in notes))
    patient_parts.append(earlier)
    return {
        role: ("\n\n".join(part for part in parts if part) + "\n").encode("utf-8")
        for role, parts in (("therapist", therapist_parts), ("patient", patient_parts))
    }


def build_files():
    templates = {role: (HERE / "templates" / f"{role}.txt").read_text(encoding="utf-8").strip()
                 for role in ("therapist", "patient")}
    files, records = {}, []
    for path in sorted((HERE / "cases").glob("*.json")):
        try:
            case = read_json(path)
            case_id = identifier(case["id"])
            if case_id != path.stem:
                raise ValueError("Case ID must match its filename.")
            scenario_id = identifier(case["scenario"])
            scenario = read_json(HERE / "scenarios" / f"{scenario_id}.json")
            if type(scenario["max_rounds"]) is not int or scenario["max_rounds"] < 1:
                raise ValueError("max_rounds must be a positive integer.")
            record = {"id": case_id, "category": identifier(scenario["category"]),
                      "mode": scenario["mode"], "max_rounds": scenario["max_rounds"]}
            for role, content in render(case, scenario, templates).items():
                files[Path(case_id) / f"{role}.txt"] = content
                record[role + "_sha256"] = hashlib.sha256(content).hexdigest()
            records.append(record)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: {error}") from error
    if not records:
        raise ValueError("No source cases were found.")
    files[Path("manifest.json")] = (json_text({"version": VERSION, "instances": records}) + "\n").encode("utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify published files without writing.")
    args = parser.parse_args()
    files = build_files()
    actual = {p.relative_to(OUTPUT) for p in OUTPUT.rglob("*") if p.is_file() and p.name != ".DS_Store"}
    extras = actual - files.keys()
    if extras:
        sys.exit("Review unexpected files under instances/: " + ", ".join(map(str, sorted(extras))))
    for name, content in files.items():
        path = OUTPUT / name
        if path.is_file() and path.read_bytes() == content:
            continue
        if args.check:
            sys.exit(f"Missing or outdated {path}. Run python construct_ins/build.py.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(f"{'Verified' if args.check else 'Built'} {(len(files) - 1) // 2} instances in {OUTPUT}")


if __name__ == "__main__":
    main()
