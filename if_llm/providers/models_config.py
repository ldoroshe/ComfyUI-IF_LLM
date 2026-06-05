from typing import Any, Dict, List, Optional

# Type alias for provider configuration dictionary
ProviderConfig = Dict[str, Any]

_MODEL_PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "fallback_models": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
            "gpt-3.5-turbo", "dall-e-3", "dall-e-2",
            "whisper-1", "tts-1", "text-embedding-3-large",
            "o1", "o1-mini", "o1-preview", "o3-mini",
        ],
        "key_env_var": "OPENAI_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "xai": {
        "url": "https://api.x.ai/v1/models",
        "fallback_models": [
            "grok-2", "grok-2-1212", "grok-2-latest",
            "grok-2-vision", "grok-2-vision-latest",
            "grok-beta", "grok-vision-beta",
        ],
        "key_env_var": "XAI_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/models",
        "fallback_models": [
            "codestral-latest", "ministral-8b-latest",
            "mistral-large-latest", "mistral-small-latest",
            "open-mistral-nemo", "pixtral-large-latest",
        ],
        "key_env_var": "MISTRAL_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "fallback_models": [
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile", "llama-3.2-11b-vision-preview",
            "whisper-large-v3", "gemma2-9b-it",
        ],
        "key_env_var": "GROQ_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/models",
        "fallback_models": ["deepseek-reasoner", "deepseek-chat", "deepseek-coder"],
        "key_env_var": "DEEPSEEK_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "anthropic": {
        "url": None,
        "fallback_models": [
            "claude-3-5-opus-latest", "claude-3-5-sonnet-latest",
            "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "claude-3-5-haiku-latest",
        ],
        "key_env_var": "ANTHROPIC_API_KEY",
        "is_local": False,
        "parse_fn": None,
    },
    "gemini": {
        "url": None,
        "fallback_models": [
            "gemini-2.0-flash-exp", "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest", "gemini-pro",
        ],
        "key_env_var": "GOOGLE_API_KEY",
        "is_local": False,
        "parse_fn": None,
    },
    "ollama": {
        "url_template": "http://{base_ip}:{port}/api/tags",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: [m["name"] for m in data.get("models", [])],
    },
    "lmstudio": {
        "url_template": "http://{base_ip}:{port}/v1/models",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "llamacpp": {
        "url_template": "http://{base_ip}:{port}/v1/models",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "vllm": {
        "url_template": "http://{base_ip}:{port}/v1/models",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: [m["id"] for m in data.get("data", [])],
    },
    "textgen": {
        "url_template": "http://{base_ip}:{port}/v1/internal/model/list",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: data.get("model_names", []),
    },
    "kobold": {
        "url_template": "http://{base_ip}:{port}/api/v1/model",
        "fallback_models": [],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": lambda data: [data.get("result", "")],
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/framework/all",
        "fallback_models": [
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "Qwen/Qwen2-VL-7B-Chat", "stabilityai/sdxl-turbo",
            "black-forest-labs/FLUX.1-dev", "gpt2",
        ],
        "key_env_var": "HUGGINGFACE_API_KEY",
        "is_local": False,
        "parse_fn": lambda data: [
            m["model_id"]
            for framework in data
            for m in framework.get("models", [])
        ],
    },
    "transformers": {
        "url": None,
        "fallback_models": [
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "Qwen/QwQ-32B-AWQ",
        ],
        "key_env_var": None,
        "is_local": True,
        "parse_fn": None,
    },
}


def get_provider_config(provider: str) -> Optional[ProviderConfig]:
    """Get configuration for a provider.

    Args:
        provider: The provider name (e.g., "openai", "ollama").

    Returns:
        The provider configuration dict, or None if not found.
    """
    return _MODEL_PROVIDERS.get(provider)


def get_all_providers() -> List[str]:
    """Get list of all configured provider names.

    Returns:
        A list of all provider name strings.
    """
    return list(_MODEL_PROVIDERS.keys())


def get_fallback_models(provider: str) -> List[str]:
    """Get fallback model list for a provider.

    Args:
        provider: The provider name.

    Returns:
        A list of fallback model names for the provider.
    """
    config = _MODEL_PROVIDERS.get(provider)
    return config["fallback_models"] if config else []


def needs_api_key(provider: str) -> bool:
    """Check if a provider requires an API key.

    Args:
        provider: The provider name.

    Returns:
        True if the provider requires an API key, False otherwise.
    """
    config = _MODEL_PROVIDERS.get(provider)
    return config is not None and config["key_env_var"] is not None


def is_local_provider(provider: str) -> bool:
    """Check if a provider is local/self-hosted.

    Args:
        provider: The provider name.

    Returns:
        True if the provider is local/offline, False otherwise.
    """
    config = _MODEL_PROVIDERS.get(provider)
    return config["is_local"] if config else False
