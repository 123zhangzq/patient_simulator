"""Run two prompt-defined agents sequentially on one local Qwen model on CUDA."""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from settings import DEFAULT_CONFIG, load_config, read_prompt, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalQwen:
    def __init__(self, config: dict):
        if platform.system() != "Linux":
            raise RuntimeError("本项目面向 Linux GPU 服务器，请在 Linux 上运行。")
        if not torch.cuda.is_available():
            raise RuntimeError("未发现可用 CUDA GPU，请检查 NVIDIA 驱动和 CUDA 版 PyTorch。")
        self.config = config
        if config["dtype"] == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = getattr(torch, config["dtype"])
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16，请将 dtype 改为 float16。")

        local_options = {"local_files_only": True, "trust_remote_code": False}
        model_config = AutoConfig.from_pretrained(config["model_dir"], **local_options)
        if model_config.model_type == "qwen3_5":
            # The original 9B checkpoint includes a vision encoder. Load its native
            # class but supply text only; no image processor or vision inputs needed.
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        elif model_config.model_type == "qwen3":
            model_class = AutoModelForCausalLM
        else:
            raise ValueError("此最小版本仅支持 Qwen3.5 和 Qwen3 的上述配置。")

        text_config = getattr(model_config, "text_config", model_config)
        model_limit = getattr(text_config, "max_position_embeddings", None)
        if model_limit and config["max_context_tokens"] > model_limit:
            raise ValueError(f"max_context_tokens 超过模型上下文限制 {model_limit}。")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], **local_options)
        print(f"加载本地模型到 cuda:0，dtype={self.dtype}（仅加载一次）……", flush=True)
        self.model = model_class.from_pretrained(
            config["model_dir"],
            config=model_config,
            dtype=self.dtype,
            device_map={"": "cuda:0"},
            **local_options,
        ).eval()

    @torch.inference_mode()
    def reply(self, messages: list[dict]) -> dict:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        ).to("cuda:0")
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens + self.config["max_new_tokens"] > self.config["max_context_tokens"]:
            raise ValueError(
                f"对话需要 {input_tokens} 个输入 token 和最多 "
                f"{self.config['max_new_tokens']} 个输出 token，超过配置的 "
                f"{self.config['max_context_tokens']}。请减少轮数/回复长度，"
                "或在显存允许时提高 max_context_tokens。历史没有被裁剪。"
            )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"],
            repetition_penalty=self.config["repetition_penalty"],
            pad_token_id=pad_id,
            use_cache=True,
        )
        generated = outputs[0, input_tokens:]
        content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not content:
            raise RuntimeError("模型生成了空回复，本次实验已停止，请检查 prompt 和生成参数。")
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        ended_with_eos = bool(eos_ids and generated[-1].item() in eos_ids)
        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": generated.numel(),
            "finish_reason": "eos" if ended_with_eos else "length",
        }


def run(config: dict, config_path: Path) -> Path:
    model_dir = Path(config["model_dir"])
    manifest_path = model_dir / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"没有完整下载记录：{manifest_path}。请先运行 "
            f"python download_model.py --config {config_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["model_id"] != config["model_id"]
            or manifest["requested_revision"] != config["revision"]):
        raise ValueError("本地模型下载记录与配置的 model_id/revision 不一致，请先重新下载。")
    prompts = {
        "therapist": read_prompt(config["therapist_prompt"]),
        "patient": read_prompt(config["patient_prompt"]),
        "opening": read_prompt(config["opening_prompt"]),
    }
    # Each agent's own messages are assistant; its counterpart's messages are user.
    # The therapist never receives the patient's private system prompt.
    histories = {
        "Therapist": [
            {"role": "system", "content": prompts["therapist"]},
            {"role": "user", "content": prompts["opening"]},
        ],
        "Patient": [{"role": "system", "content": prompts["patient"]}],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path(config["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "started_at_utc": utc_now(),
        "completed_rounds": 0, "utterances": 0,
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
    print(f"本次实验输出：{output_dir}", flush=True)
    try:
        set_seed(config["seed"])
        llm = LocalQwen(config)
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["dtype"] = str(llm.dtype)
        write_json(metadata_path, metadata)
        with (output_dir / "dialogue.jsonl").open("w", encoding="utf-8") as jsonl, (
            output_dir / "dialogue.md"
        ).open("w", encoding="utf-8") as markdown:
            markdown.write(f"# Therapist / Patient\n\n模型：{config['model_id']}\n\n")
            markdown.flush()
            for round_number in range(1, config["rounds"] + 1):
                for speaker, listener in (("Therapist", "Patient"), ("Patient", "Therapist")):
                    result = llm.reply(histories[speaker])
                    record = {
                        "round": round_number, "speaker": speaker,
                        "created_at_utc": utc_now(), **result,
                    }
                    # Persist every utterance before starting the next generation.
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    markdown.write(f"## 第 {round_number} 轮 · {speaker}\n\n{result['content']}\n\n")
                    markdown.flush()
                    histories[speaker].append({"role": "assistant", "content": result["content"]})
                    histories[listener].append({"role": "user", "content": result["content"]})
                    metadata["utterances"] += 1
                    if speaker == "Patient":
                        metadata["completed_rounds"] = round_number
                    write_json(metadata_path, metadata)
                    print(f"\n[第 {round_number} 轮] {speaker}: {result['content']}", flush=True)
                    if result["finish_reason"] == "length":
                        print("提示：本条回复达到 max_new_tokens，可能未说完。", flush=True)
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
    print(f"\n已完成 {config['rounds']} 轮、{metadata['utterances']} 条发言。", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Linux CUDA GPU 上离线运行双角色 Qwen 对话。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, help="覆盖配置中的对话轮数；一轮包含双方各一次发言。")
    parser.add_argument("--seed", type=int, help="覆盖配置中的随机种子。")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds 必须是正整数。")
        config["rounds"] = args.rounds
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed 必须是 0 到 2**32-1 之间的整数。")
        config["seed"] = args.seed
    run(config, config_path)


if __name__ == "__main__":
    main()
