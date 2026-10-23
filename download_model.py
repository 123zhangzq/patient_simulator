"""Download model files once; this entry point does not load weights or use a GPU."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from settings import DEFAULT_CONFIG, load_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Qwen 到本地，供后续离线实验使用。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    model_dir = Path(config["model_dir"])
    manifest_path = model_dir / "download_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["model_id"] != config["model_id"]:
            raise ValueError("model_dir 中已有其他模型，请为新模型选择另一个目录。")

    # Resolve a branch/tag once, so every file comes from the same immutable commit.
    # Authentication, if needed, comes from HF_TOKEN or `hf auth login`.
    info = HfApi().model_info(config["model_id"], revision=config["revision"])
    if not info.sha:
        raise RuntimeError("无法解析模型的 Hugging Face commit。")
    print(f"下载 {config['model_id']} @ {info.sha} 到 {model_dir}", flush=True)
    # A failed update must not leave an old success marker next to partial files.
    manifest_path.unlink(missing_ok=True)
    snapshot_download(
        repo_id=config["model_id"],
        revision=info.sha,
        local_dir=str(model_dir),
    )
    write_json(manifest_path, {
        "model_id": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": info.sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("下载完成。后续直接运行 run_dialogue.py，无需再次下载。", flush=True)


if __name__ == "__main__":
    main()
