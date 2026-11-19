"""Select a text model without importing unused SDKs or local inference libraries."""


def load_provider(model_config: dict):
    provider = model_config["provider"]
    if provider == "openrouter":
        from .openrouter import OpenRouter

        return OpenRouter(model_config)
    if provider == "local_qwen":
        from .local_qwen import LocalQwen

        return LocalQwen(model_config)
    raise ValueError(f"Unsupported provider: {provider}")
