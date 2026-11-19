"""Run a standardized patient session with one prompt per role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import DEFAULT_CONFIG, PROJECT_DIR, load_config, validate_roles
from core.instances import load_instances


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instances-dir", type=Path, default=PROJECT_DIR / "instances")
    parser.add_argument("--instance", action="append", metavar="ID", help="Run this instance; repeat to select multiple.")
    parser.add_argument("--therapist", metavar="MODEL", help="Therapist model alias from config.json.")
    parser.add_argument("--patient", action="append", metavar="MODEL", help="Patient model alias; repeat to compare multiple.")
    parser.add_argument("--rounds", type=int, help="Override the maximum new rounds for this run.")
    parser.add_argument("--seed", type=int, help="Override the sampling seed.")
    parser.add_argument("--allow-edited-prompts", action="store_true", help="Run locally modified prompt text and record its actual hash.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-instances", action="store_true", help="List instances without loading a model.")
    modes.add_argument("--dry-run", action="store_true", help="Show selected models, instances and prompts without calling a model.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        roles = config["roles"]
        if args.therapist is not None:
            roles["therapist"] = args.therapist
        if args.patient is not None:
            roles["patients"] = args.patient
        validate_roles(roles, config["models"])
        if args.seed is not None:
            if not 0 <= args.seed < 2**32:
                parser.error("--seed must be an integer from 0 to 2**32 - 1.")
            config["run"]["generation"]["seed"] = args.seed
        instances = load_instances(args.instances_dir, allow_edited_prompts=args.allow_edited_prompts)
        if args.instance:
            requested = set(args.instance)
            unknown = requested - {i["id"] for i in instances}
            if unknown:
                parser.error(f"Unknown instance IDs: {', '.join(sorted(unknown))}")
            instances = [i for i in instances if i["id"] in requested]
        if args.rounds is not None:
            if args.rounds < 1:
                parser.error("--rounds must be positive.")
            for i in instances:
                i["max_rounds"] = args.rounds
        if args.list_instances:
            for i in instances:
                print(f"{i['id']}\t{i['category']}\t{i['mode']}\tup to {i['max_rounds']} rounds")
            return
        if args.dry_run:
            print(json.dumps({"roles": roles, "run": config["run"], "instances": instances}, indent=2, ensure_ascii=False))
            return
        if any(config["models"][alias]["provider"] == "openrouter" for alias in [roles["therapist"], *roles["patients"]]):
            try:
                from dotenv import load_dotenv
            except ImportError:
                parser.error("Install API dependencies with: python -m pip install -r requirements.txt")
            load_dotenv(PROJECT_DIR / ".env", override=False, interpolate=False)
        from core.runner import run
        run(config, instances)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
