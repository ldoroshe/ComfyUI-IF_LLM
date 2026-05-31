import pytest
from if_llm.providers.models_config import (
    get_provider_config, get_all_providers, get_fallback_models,
    needs_api_key, is_local_provider,
)


class TestGetProviderConfig:
    def test_known_provider(self):
        config = get_provider_config("openai")
        assert config is not None
        assert config["key_env_var"] == "OPENAI_API_KEY"

    def test_unknown_provider(self):
        assert get_provider_config("nonexistent") is None


class TestGetAllProviders:
    def test_returns_all(self):
        providers = get_all_providers()
        assert "openai" in providers
        assert "ollama" in providers
        assert len(providers) >= 13


class TestNeedsApiKey:
    def test_openai_needs_key(self):
        assert needs_api_key("openai") is True

    def test_ollama_no_key(self):
        assert needs_api_key("ollama") is False

    def test_transformers_no_key(self):
        assert needs_api_key("transformers") is False


class TestIsLocalProvider:
    def test_ollama_is_local(self):
        assert is_local_provider("ollama") is True

    def test_openai_not_local(self):
        assert is_local_provider("openai") is False

    def test_huggingface_not_local(self):
        assert is_local_provider("huggingface") is False


class TestFallbackModels:
    def test_openai_has_fallbacks(self):
        models = get_fallback_models("openai")
        assert "gpt-4o" in models

    def test_ollama_no_fallbacks(self):
        models = get_fallback_models("ollama")
        assert len(models) == 0
