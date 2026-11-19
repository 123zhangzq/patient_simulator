"""OpenRouter chat completions via the OpenAI Python SDK."""

import os


class OpenRouter:
    def __init__(self, config: dict):
        key_name = config.get("api_key_env", "OPENROUTER_API_KEY")
        key = os.environ.get(key_name, "").strip()
        if not key:
            raise ValueError(f"Set {key_name} in .env or your environment to use OpenRouter.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the API dependencies: python -m pip install -r requirements.txt") from exc

        self.model_id = config["model_id"]
        self.client = OpenAI(
            api_key=key,
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            timeout=config.get("timeout_seconds", 90.0),
            max_retries=config.get("max_retries", 2),
        )

    def reply(self, messages: list[dict], *, generation: dict) -> dict:
        from openai import OpenAIError

        kwargs = {key: generation[key] for key in ("temperature", "top_p", "seed") if key in generation}
        extra = {key: generation[key] for key in ("top_k", "repetition_penalty") if key in generation}
        try:
            result = self.client.chat.completions.create(
                model=self.model_id, messages=messages, max_tokens=generation["max_new_tokens"],
                extra_body=extra or None, **kwargs,
            )
        except OpenAIError as exc:
            # Provider error bodies can echo request data. Persist a safe diagnostic instead.
            status = getattr(exc, "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(f"OpenRouter request for {self.model_id} failed ({detail}).") from None
        if not result.choices:
            raise RuntimeError(f"OpenRouter returned no choices for {self.model_id}.")
        choice = result.choices[0]
        reason = choice.finish_reason
        if reason not in ("stop", "length"):
            raise RuntimeError(f"OpenRouter did not complete a text reply (finish_reason={reason!r}).")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenRouter returned an empty reply for {self.model_id} (finish_reason={reason}).")
        usage = result.usage
        return {"content": content.strip(), "finish_reason": reason,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "model": result.model, "request_id": result.id}
