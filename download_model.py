"""Prepare a local Qwen checkpoint once, using a model alias from config.json."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from core.config import DEFAULT_CONFIG, load_config
from core.providers.local_qwen import verify_checkpoint


def download(config: dict) -> None:
    model_dir = Path(config["model_dir"])
    manifest_path = model_dir / "download_manifest.json"
    if manifest_path.exists():
        manifest = verify_checkpoint(config)
        print(f"Reusing {manifest['model_id']} @ {manifest['resolved_revision']} in {model_dir}.")
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-local.txt before downloading Qwen.") from exc
    model_id = config["model_id"]
    revision = config["revision"]
    commit = HfApi().model_info(model_id, revision=revision).sha
    if not commit:
        raise RuntimeError("Could not resolve the requested Hugging Face revision.")
    print(f"Downloading {model_id} @ {commit} to {model_dir}", flush=True)
    snapshot_download(repo_id=model_id, revision=commit, local_dir=str(model_dir))
    manifest = {"model_id": model_id, "requested_revision": revision, "resolved_revision": commit,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
    # A manifest marks a completed snapshot; publish it only after every file is present.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=model_dir, delete=False) as tmp:
        json.dump(manifest, tmp, indent=2)
        tmp.write("\n")
    Path(tmp.name).replace(manifest_path)
    print("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default="qwen_local", help="A local_qwen model alias.")
    args = parser.parse_args()
    try:
        models = load_config(args.config)["models"]
        if args.model not in models or models[args.model]["provider"] != "local_qwen":
            parser.error(f"{args.model!r} must name a local_qwen entry in config.json.")
        download(models[args.model])
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
