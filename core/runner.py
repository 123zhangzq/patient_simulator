"""Run independent dialogue cases while sharing one loaded language model."""

import json
import copy
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import set_seed

from core.cases import (DialogueCase, compose_prompts, initial_history, load_cases,
                        session_fingerprint, session_information)
from core.interfaces import DialogueMessage
from core.patient_setup import prepare_patient_setup
from core.profiles import profile_fingerprint
from core.llm import LocalQwen
from core.registry import load_patient_class
from core.therapist import Therapist, load_control
from settings import PROJECT_DIR, read_prompt, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_case(
    llm,
    patient_class,
    prompts: dict[str, str],
    config: dict,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Persist one dialogue, including its partial progress on interruption."""
    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    try:
        # Each case starts with fresh agents, history, and an identical random
        # seed. Earlier cases cannot change this case's patient or sampling state.
        set_seed(config["seed"])
        case = DialogueCase(Path(metadata["case"]["source_file"]), metadata["case"]["data"])
        therapist = Therapist(llm, prompts["therapist"], prompts["opening"],
                              max_rounds=config["rounds"],
                              opening_utterance=case.data["therapist"]["opening_question"])
        setup = metadata["patient_setup"]
        patient = patient_class(
            llm, setup["system_prompt"],
            patient_profile=copy.deepcopy(metadata["case"]["data"]["patient_profile"]),
            seed=config["seed"], options=copy.deepcopy(setup["options"]),
            prompts=copy.deepcopy(setup["prompts"]),
        )
        public_history = list(initial_history(case))
        patient.initialize_history(tuple(public_history))
        metadata["initial_history_utterances"] = len(public_history)
        metadata["patient_initialization"] = patient.describe()
        write_json(metadata_path, metadata)
        write_json(output_dir / "initial_history.json", session_information(case)["initial_history"])
        with (output_dir / "dialogue.jsonl").open("w", encoding="utf-8") as jsonl, (
            output_dir / "dialogue.md"
        ).open("w", encoding="utf-8") as markdown, (
            output_dir / "patient_trace.jsonl"
        ).open("w", encoding="utf-8") as trace_file:
            # Kept separate from public speech and from the patient-only trace.
            with (output_dir / "therapist_trace.jsonl").open("w", encoding="utf-8") as therapist_trace_file:
                _exchange(patient, therapist, public_history, config, metadata, metadata_path,
                          jsonl, markdown, trace_file, therapist_trace_file)
        metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["ended_at_utc"] = utc_now()
        write_json(metadata_path, metadata)


def _exchange(patient, therapist, public_history, config, metadata, metadata_path,
              jsonl, markdown, trace_file, therapist_trace_file):
    """Generate new exchanges after the supplied, separately recorded prefix."""
    markdown.write(f"# Therapist / Patient\n\nModel: {config['model_id']}\n\n")
    if public_history:
        markdown.write("## Supplied earlier conversation (not generated or scored)\n\n")
        for message in public_history:
            markdown.write(f"**{message.speaker}:** {message.content}\n\n")
        markdown.write("## New dialogue\n\n")
    markdown.flush()
    for round_number in range(1, config["rounds"] + 1):
        close_reason = None
        for speaker, agent in (("Therapist", therapist), ("Patient", patient)):
            # Only supplied prefix speech and newly spoken text enter history.
            completed = False
            agent.last_trace = None
            try:
                result = agent.respond(tuple(public_history))
                content = result["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{speaker} returned an empty or invalid reply.")
                completed = True
            finally:
                # Internal state and model inputs never enter public history.
                if agent.last_trace is not None:
                    destination = trace_file if speaker == "Patient" else therapist_trace_file
                    destination.write(json.dumps({
                        "round": round_number,
                        "status": "completed" if completed else "failed",
                        "trace": agent.last_trace,
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                    destination.flush()
            if speaker == "Therapist" and result.get("dialogue_action") == "close":
                close_reason = result.get("stop_reason") or "activity_complete"
            record = {
                "round": round_number, "speaker": speaker,
                "created_at_utc": utc_now(),
                "content": content,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "finish_reason": result["finish_reason"],
            }
            # Persist every utterance before starting the next generation.
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()
            markdown.write(f"## Round {round_number} - {speaker}\n\n{content}\n\n")
            markdown.flush()
            public_history.append(DialogueMessage(speaker=speaker, content=content))
            metadata["utterances"] += 1
            if speaker == "Patient":
                metadata["completed_rounds"] = round_number
            write_json(metadata_path, metadata)
            print(f"\n[Round {round_number}] {speaker}: {content}", flush=True)
            if result["finish_reason"] == "length":
                print("Note: This reply reached max_new_tokens and may be incomplete.", flush=True)
        if close_reason:
            metadata["stop_reason"] = close_reason
            break
    else:
        metadata["stop_reason"] = "round_limit"


def run(
    config: dict,
    config_path: Path,
    patient_method: str = "single_prompt",
    cases: Sequence[DialogueCase] | None = None,
    rounds_override: int | None = None,
) -> Path:
    """Run cases in order, using all repository cases when none are supplied.

    Every method receives the same case-contained profile. Source workbooks and
    reference transcripts are not read by the experiment.
    """
    patient_class = load_patient_class(patient_method)
    control = load_control()
    model_dir = Path(config["model_dir"])
    manifest_path = model_dir / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No completed model download found at {manifest_path}. First run: "
            f"python download_model.py --config {config_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["model_id"] != config["model_id"]
            or manifest["requested_revision"] != config["revision"]):
        raise ValueError(
            "The local download manifest does not match the configured model_id/revision. "
            "Run the download script with this configuration first."
        )
    base_prompts = {
        "therapist": read_prompt(config["therapist_prompt"]),
        "opening": read_prompt(config["opening_prompt"]),
    }
    selected_cases = list(cases) if cases is not None else load_cases(PROJECT_DIR / "cases")
    if not selected_cases:
        raise ValueError("At least one dialogue case is required.")
    if len({case.case_id for case in selected_cases}) != len(selected_cases):
        raise ValueError("Dialogue case IDs must be unique within a batch.")
    prepared = []
    for case in selected_cases:
        prompts = dict(base_prompts)
        composed = compose_prompts(case, base_prompts["therapist"], base_prompts["opening"])
        # Deliberately select only therapist inputs: a case cannot override
        # the configured patient method, prompt, or generated conversation.
        prompts["therapist"] = composed["therapist"]
        prompts["opening"] = composed["opening"]
        setup = prepare_patient_setup(config, patient_method, case.data["patient_profile"],
                                      session_information(case)["shared_context"])
        prompts["patient"] = "\n\n".join(part for part in
            (setup["prompts"].get("response", ""), setup["system_prompt"]) if part)
        rounds = rounds_override if rounds_override is not None else (
            case.data.get("rounds", config["rounds"])
        )
        if type(rounds) is not int or rounds <= 0:
            raise ValueError("The number of dialogue rounds must be a positive integer.")
        prepared.append((case, prompts, {**config, "rounds": rounds}, setup))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path(config["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "patient_method": patient_method,
        "config_file": str(config_path), "config": config,
        "model_download": manifest,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": version("accelerate"),
            "huggingface_hub": version("huggingface_hub"),
            "safetensors": version("safetensors"),
        },
    }
    batch = {
        **common, "status": "running", "batch_status": "running",
        "started_at_utc": utc_now(), "case_count": len(prepared),
        "completed_cases": 0,
        "cases": [
            {
                "case_id": case.case_id,
                "output_dir": str(output_dir / case.case_id),
                "status": "pending", "effective_rounds": case_config["rounds"],
                "completed_rounds": 0, "utterances": 0,
            }
            for case, _, case_config, _ in prepared
        ],
    }
    batch_path = output_dir / "batch_metadata.json"
    write_json(batch_path, batch)
    print(f"Experiment output directory: {output_dir}", flush=True)
    try:
        set_seed(config["seed"])
        llm = LocalQwen(config)
        common["gpu"] = batch["gpu"] = torch.cuda.get_device_name(0)
        common["dtype"] = batch["dtype"] = str(llm.dtype)
        write_json(batch_path, batch)
        for index, (case, prompts, case_config, setup) in enumerate(prepared):
            case_dir = output_dir / case.case_id
            case_dir.mkdir(exist_ok=False)
            metadata = {
                **common, "status": "running", "started_at_utc": utc_now(),
                "completed_rounds": 0, "utterances": 0,
                "effective_rounds": case_config["rounds"],
                "effective_config": case_config,
                "prompts": prompts,
                "patient_setup": setup,
                "patient_profile_sha256": profile_fingerprint(case.data["patient_profile"]),
                "session_sha256": session_fingerprint(case),
                "therapist_control": control,
                "case": {"source_file": str(case.path), "data": case.data},
                "case_id": case.case_id,
            }
            entry = batch["cases"][index]
            entry["status"] = "running"
            write_json(batch_path, batch)
            print(f"\nCase {index + 1}/{len(prepared)}: {case.case_id}", flush=True)
            try:
                _run_case(llm, patient_class, prompts, case_config, case_dir, metadata)
            finally:
                # _run_case records its final status before it propagates errors.
                # This summary therefore includes completed partial rounds, too.
                for field in ("status", "completed_rounds", "utterances", "stop_reason", "error", "ended_at_utc"):
                    if field in metadata:
                        entry[field] = metadata[field]
                write_json(batch_path, batch)
            batch["completed_cases"] += 1
            write_json(batch_path, batch)
            print(
                f"\nCompleted {metadata['completed_rounds']} rounds and "
                f"{metadata['utterances']} utterances.", flush=True,
            )
        batch["status"] = batch["batch_status"] = "completed"
    except KeyboardInterrupt:
        batch["status"] = batch["batch_status"] = "interrupted"
        raise
    except Exception as error:
        batch["status"] = batch["batch_status"] = "failed"
        batch["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        batch["ended_at_utc"] = utc_now()
        write_json(batch_path, batch)
    return output_dir
