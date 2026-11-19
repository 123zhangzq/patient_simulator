"""Offline text generation using an existing Qwen3 or Qwen3.5 checkpoint."""

import json
from pathlib import Path


def read_manifest(model_dir: Path) -> dict:
    path = model_dir / "download_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No completed download at {path}. Run download_model.py --model <alias> "
            "or point model_dir at an existing benchmark checkpoint."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not all(manifest.get(key) for key in (
        "model_id", "requested_revision", "resolved_revision",
    )):
        raise ValueError(f"Incomplete download manifest: {path}")
    return manifest


def verify_checkpoint(config: dict) -> dict:
    manifest = read_manifest(Path(config["model_dir"]))
    if (manifest["model_id"] != config["model_id"]
            or config["revision"] not in (manifest["requested_revision"], manifest["resolved_revision"])):
        raise ValueError("model_id or revision differs from the local download manifest. "
                         "Select the matching checkpoint or download into a separate model_dir.")
    return manifest


class LocalQwen:
    def __init__(self, config: dict):
        manifest = verify_checkpoint(config)
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install CUDA PyTorch/torchvision and requirements-local.txt for local Qwen.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Local Qwen requires a CUDA GPU and CUDA-enabled PyTorch.")
        dtype = config.get("dtype", "auto")
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("The GPU does not support BF16. Use dtype=float16.")

        model_dir = config["model_dir"]
        options = {"local_files_only": True, "trust_remote_code": False}
        model_config = AutoConfig.from_pretrained(model_dir, **options)
        if model_config.model_type == "qwen3_5":
            # Keep the native class for checkpoints with a vision encoder; input is text only.
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        elif model_config.model_type == "qwen3":
            model_class = AutoModelForCausalLM
        else:
            raise ValueError(f"Unsupported local model type: {model_config.model_type}")

        self.model_id = config["model_id"]
        self.revision = manifest["resolved_revision"]
        self.max_context_tokens = config["max_context_tokens"]
        text_config = getattr(model_config, "text_config", model_config)
        max_positions = getattr(text_config, "max_position_embeddings", None)
        if max_positions and self.max_context_tokens > max_positions:
            raise ValueError(f"max_context_tokens exceeds the checkpoint's limit of {max_positions}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **options)
        self.model = model_class.from_pretrained(
            model_dir, config=model_config, dtype=getattr(torch, dtype),
            device_map={"": "cuda:0"}, **options,
        ).eval()

    def reply(self, messages: list[dict], *, generation: dict) -> dict:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        input_tokens = inputs["input_ids"].shape[-1]
        budget = generation["max_new_tokens"]
        if input_tokens + budget > self.max_context_tokens:
            raise ValueError(f"Input ({input_tokens}) + reply budget ({budget}) exceed "
                             f"max_context_tokens ({self.max_context_tokens}).")
        do_sample = generation.get("temperature", 0.7) > 0
        kwargs = {key: generation[key] for key in ("temperature", "top_p", "top_k")
                  if do_sample and key in generation}
        if "repetition_penalty" in generation:
            kwargs["repetition_penalty"] = generation["repetition_penalty"]
        if "seed" in generation and generation["seed"] is not None:
            torch.manual_seed(generation["seed"])
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=budget, do_sample=do_sample, pad_token_id=pad_id,
                use_cache=True, **kwargs,
            )
        generated = outputs[0, input_tokens:]
        content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not content:
            raise RuntimeError(f"Local model {self.model_id} returned an empty reply.")
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        ended = bool(eos_ids and generated[-1].item() in eos_ids)
        return {"content": content, "finish_reason": "stop" if ended else "length",
                "input_tokens": input_tokens, "output_tokens": generated.numel(),
                "model": self.model_id, "revision": self.revision}
