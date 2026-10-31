"""Load one local Qwen model and generate text replies on a Linux CUDA GPU."""

import platform

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core.interfaces import GenerationResult


class LocalQwen:
    def __init__(self, config: dict):
        if platform.system() != "Linux":
            raise RuntimeError("This project requires a Linux GPU server. Please run it on Linux.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU is available. Check the NVIDIA driver and CUDA-enabled PyTorch installation."
            )
        self.config = config
        if config["dtype"] == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = getattr(torch, config["dtype"])
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The current GPU does not support BF16. Set dtype to float16.")

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
            raise ValueError(
                "This minimal version only supports the provided Qwen3.5 and Qwen3 configurations."
            )

        text_config = getattr(model_config, "text_config", model_config)
        model_limit = getattr(text_config, "max_position_embeddings", None)
        if model_limit and config["max_context_tokens"] > model_limit:
            raise ValueError(f"max_context_tokens exceeds the model's context limit of {model_limit}.")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], **local_options)
        print(
            f"Loading the local model on cuda:0 with dtype={self.dtype} (loaded once per run)...",
            flush=True,
        )
        self.model = model_class.from_pretrained(
            config["model_dir"],
            config=model_config,
            dtype=self.dtype,
            device_map={"": "cuda:0"},
            **local_options,
        ).eval()

    @torch.inference_mode()
    def reply(self, messages: list[dict]) -> GenerationResult:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        ).to("cuda:0")
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens + self.config["max_new_tokens"] > self.config["max_context_tokens"]:
            raise ValueError(
                f"The dialogue requires {input_tokens} input tokens and up to "
                f"{self.config['max_new_tokens']} output tokens, exceeding the configured "
                f"max_context_tokens limit of {self.config['max_context_tokens']}. "
                "Reduce the number of rounds or reply length, or increase max_context_tokens "
                "if GPU memory allows. The conversation history has not been truncated."
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
            raise RuntimeError(
                "The model generated an empty reply. The experiment has stopped; "
                "check the prompts and generation settings."
            )
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
