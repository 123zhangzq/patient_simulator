"""Command-line entry point for the shared Therapist-Patient dialogue experiment."""

import argparse
from pathlib import Path

from core.registry import available_patient_methods
from settings import DEFAULT_CONFIG, load_config


def main() -> None:
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
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds must be a positive integer.")
        config["rounds"] = args.rounds
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must be an integer from 0 to 2**32 - 1.")
        config["seed"] = args.seed

    # Keep command-line help independent of the installed GPU runtime.
    from core.runner import run

    run(config, config_path, patient_method=args.patient)


if __name__ == "__main__":
    main()
