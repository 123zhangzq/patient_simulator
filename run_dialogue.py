"""Command-line entry point for the shared Therapist-Patient dialogue experiment."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from core.cases import compose_prompts, load_cases, select_cases, session_fingerprint, session_information
from core.registry import available_patient_methods
from core.patient_setup import prepare_patient_setup
from core.profiles import patient_information, profile_fingerprint
from core.therapist import load_control
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
        help="Override the maximum new dialogue rounds; each round has one reply per agent.",
    )
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument("--no-attention", action="store_true", help="Disable attention filtering for patient_agents.")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory impairment for patient_agents.")
    parser.add_argument("--interaction", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable patient interaction annotations without changing profile or memory settings.")
    parser.add_argument("--show-patient", action="store_true", help="Include private patient information and instructions in --dry-run output.")
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
    if args.show_patient and not args.dry_run:
        parser.error("--show-patient requires --dry-run.")
    if (args.no_attention or args.no_memory) and args.patient != "patient_agents":
        parser.error("--no-attention and --no-memory require --patient patient_agents.")
    if args.interaction is not None and args.patient != "patient_agents":
        parser.error("--interaction and --no-interaction require --patient patient_agents.")
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
        if args.no_attention or args.no_memory:
            agent_options = config.setdefault("patient_agents", {})
            if args.no_attention:
                agent_options["attention_enabled"] = False
            if args.no_memory:
                agent_options["memory_enabled"] = False
        if args.interaction is not None:
            config.setdefault("patient_agents", {})["interaction_enabled"] = args.interaction
        cases = select_cases(load_cases(args.cases_dir), case_ids=args.case_ids, category=args.category)
        if args.list_cases:
            for case in cases:
                rounds = args.rounds if args.rounds is not None else case.data.get("rounds", config["rounds"])
                print(f"{case.case_id}\t{case.data['category']}\tup to {rounds} rounds\t{case.data['title']}")
            return
        therapist_template = read_prompt(config["therapist_prompt"])
        opening_template = read_prompt(config["opening_prompt"])
        control = load_control()
        # Fail before loading weights if any selected case or template is invalid.
        prepared = [(case, compose_prompts(case, therapist_template, opening_template)) for case in cases]
        patient_setups = {
            case.case_id: prepare_patient_setup(config, args.patient, case.data["patient_profile"],
                                               session_information(case)["shared_context"])
            for case in cases
        }
        if args.dry_run:
            print(json.dumps({
                "config_file": str(config_path), "patient_method": args.patient,
                "patient_prompt_file": (config["patient_prompt"] if args.patient == "single_prompt" else
                    config.get("patient_agents", {}).get("response_prompt", str(PROJECT_DIR / "prompts/patient_agents/response.txt"))),
                "seed": config["seed"],
                "therapist_control": control,
                "cases": [{
                    "id": case.case_id, "category": case.data["category"],
                    "case_file": str(case.path),
                    "rounds": args.rounds if args.rounds is not None else case.data.get("rounds", config["rounds"]),
                    "prompts": prompts,
                    "patient_id": case.data["patient_profile"]["participant_id"],
                    "patient_profile_sha256": profile_fingerprint(case.data["patient_profile"]),
                    "session_sha256": session_fingerprint(case),
                    "session": session_information(case),
                    **({"patient_information": patient_information(case.data["patient_profile"]),
                        "patient_setup": patient_setups[case.case_id]} if args.show_patient else {}),
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
