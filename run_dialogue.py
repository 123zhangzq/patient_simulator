"""Command-line entry point for the shared Therapist-Patient dialogue experiment."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from core.cases import compose_prompts, load_cases, select_cases
from core.registry import available_patient_methods
from settings import DEFAULT_CONFIG, PROJECT_DIR, load_config, read_prompt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an offline two-agent Qwen dialogue on a Linux CUDA GPU.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--patient", choices=available_patient_methods(), default="single_prompt",
        help="Patient method to run (default: single_prompt).",
    )
    parser.add_argument(
        "--rounds", type=int,
        help="Override the number of dialogue rounds; each round has one reply per agent.",
    )
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument(
        "--cases-dir", type=Path, default=PROJECT_DIR / "cases",
        help="Directory searched recursively for case.json, or one case JSON file.",
    )
    parser.add_argument("--case", action="append", dest="case_ids", metavar="ID", help="Run one case id; repeat to select several.")
    parser.add_argument("--category", help="Run only cases with this category.")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--list-cases", action="store_true", help="List selected cases without loading a model.")
    preview.add_argument("--dry-run", action="store_true", help="Preview composed therapist prompts as JSON without loading a model.")
    args = parser.parse_args(argv)
    if args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds must be a positive integer.")
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must be an integer from 0 to 2**32 - 1.")
    try:
        config, config_path = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        cases = select_cases(load_cases(args.cases_dir), case_ids=args.case_ids, category=args.category)
        if args.list_cases:
            for case in cases:
                rounds = args.rounds if args.rounds is not None else case.data.get("rounds", config["rounds"])
                print(f"{case.case_id}\t{case.data['category']}\t{rounds} rounds\t{case.data['title']}")
            return
        therapist_template = read_prompt(config["therapist_prompt"])
        opening_template = read_prompt(config["opening_prompt"])
        read_prompt(config["patient_prompt"])
        # Fail before loading weights if any selected case or template is invalid.
        prepared = [(case, compose_prompts(case, therapist_template, opening_template)) for case in cases]
        if args.dry_run:
            print(json.dumps({
                "config_file": str(config_path), "patient_method": args.patient,
                "patient_prompt_file": config["patient_prompt"], "seed": config["seed"],
                "cases": [{
                    "id": case.case_id, "category": case.data["category"],
                    "case_file": str(case.path),
                    "rounds": args.rounds if args.rounds is not None else case.data.get("rounds", config["rounds"]),
                    "prompts": prompts,
                } for case, prompts in prepared],
            }, ensure_ascii=False, indent=2))
            return
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # Keep command-line help independent of the installed GPU runtime.
    from core.runner import run

    run(config, config_path, patient_method=args.patient, cases=cases, rounds_override=args.rounds)


if __name__ == "__main__":
    main()
