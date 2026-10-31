"""Run the shared dialogue protocol for a fixed therapist and a selected patient."""

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import set_seed

from core.interfaces import DialogueMessage
from core.llm import LocalQwen
from core.registry import load_patient_class
from core.therapist import Therapist
from settings import read_prompt, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(config: dict, config_path: Path, patient_method: str = "single_prompt") -> Path:
    patient_class = load_patient_class(patient_method)
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
    prompts = {
        "therapist": read_prompt(config["therapist_prompt"]),
        "patient": read_prompt(config["patient_prompt"]),
        "opening": read_prompt(config["opening_prompt"]),
    }
    public_history: list[DialogueMessage] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path(config["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "started_at_utc": utc_now(),
        "completed_rounds": 0, "utterances": 0,
        "patient_method": patient_method,
        "config_file": str(config_path), "config": config, "prompts": prompts,
        "model_download": manifest,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": version("accelerate"),
            "huggingface_hub": version("huggingface_hub"),
            "safetensors": version("safetensors"),
        },
    }
    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    print(f"Experiment output directory: {output_dir}", flush=True)
    try:
        # Preserve the baseline's single global random stream and call order.
        set_seed(config["seed"])
        llm = LocalQwen(config)
        therapist = Therapist(llm, prompts["therapist"], prompts["opening"])
        patient = patient_class(llm, prompts["patient"])
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["dtype"] = str(llm.dtype)
        write_json(metadata_path, metadata)
        with (output_dir / "dialogue.jsonl").open("w", encoding="utf-8") as jsonl, (
            output_dir / "dialogue.md"
        ).open("w", encoding="utf-8") as markdown:
            markdown.write(f"# Therapist / Patient\n\nModel: {config['model_id']}\n\n")
            markdown.flush()
            for round_number in range(1, config["rounds"] + 1):
                for speaker, agent in (("Therapist", therapist), ("Patient", patient)):
                    # Agents receive an immutable snapshot of public utterances.
                    # Their private prompts and state remain inside their objects.
                    result = agent.respond(tuple(public_history))
                    content = result["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError(f"{speaker} returned an empty or invalid reply.")
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
    print(f"\nCompleted {config['rounds']} rounds and {metadata['utterances']} utterances.", flush=True)
    return output_dir
