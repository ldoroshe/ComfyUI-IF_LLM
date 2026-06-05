"""Tests for centralized constants in constants.py."""

from if_llm.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_P,
    EMBEDDING_PROVIDERS,
    ERROR_PREFIX,
    GEMINI_MODEL_ROLE,
    IMAGE_DATA_URL_PREFIX,
    IMAGE_FORMATS,
    IMAGE_TYPE_BASE64,
    IMAGE_TYPE_IMAGE,
    IMAGE_TYPE_IMAGE_URL,
    IMAGE_TYPE_TEXT,
    LOCAL_PROVIDERS,
    MEDIA_TYPE_IMAGE_JPEG,
    MODEL_LIST_TIMEOUT,
    PROVIDER_TYPES,
    RESPONSE_KEY_CHOICES,
    RESPONSE_KEY_CONTENT,
    RESPONSE_KEY_MESSAGE,
    RESPONSE_KEY_RESPONSE,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    ImageFormat,
)


class TestProviderTypes:
    def test_contains_openai(self):
        assert "openai" in PROVIDER_TYPES

    def test_contains_ollama(self):
        assert "ollama" in PROVIDER_TYPES

    def test_is_frozenset(self):
        assert isinstance(PROVIDER_TYPES, frozenset)

    def test_has_15_providers(self):
        assert len(PROVIDER_TYPES) == 15


class TestLocalProviders:
    def test_ollama_is_local(self):
        assert "ollama" in LOCAL_PROVIDERS

    def test_openai_not_local(self):
        assert "openai" not in LOCAL_PROVIDERS

    def test_is_frozenset(self):
        assert isinstance(LOCAL_PROVIDERS, frozenset)


class TestEmbeddingProviders:
    def test_ollama_supports_embeddings(self):
        assert "ollama" in EMBEDDING_PROVIDERS

    def test_openai_supports_embeddings(self):
        assert "openai" in EMBEDDING_PROVIDERS

    def test_is_frozenset(self):
        assert isinstance(EMBEDDING_PROVIDERS, frozenset)


class TestImageFormats:
    def test_contains_openai(self):
        assert "openai" in IMAGE_FORMATS

    def test_contains_anthropic(self):
        assert "anthropic" in IMAGE_FORMATS

    def test_is_frozenset(self):
        assert isinstance(IMAGE_FORMATS, frozenset)


class TestImageFormatEnum:
    def test_openai_value(self):
        assert ImageFormat.OPENAI.value == "openai"

    def test_ollama_value(self):
        assert ImageFormat.OLLAMA.value == "ollama"

    def test_anthropic_value(self):
        assert ImageFormat.ANTHROPIC.value == "anthropic"

    def test_gemini_value(self):
        assert ImageFormat.GEMINI.value == "gemini"

    def test_is_string_enum(self):
        assert isinstance(ImageFormat.OPENAI, str)

    def test_iteration(self):
        values = [e.value for e in ImageFormat]
        assert set(values) == {"openai", "ollama", "anthropic", "gemini"}


class TestResponseKeys:
    def test_choices_key(self):
        assert RESPONSE_KEY_CHOICES == "choices"

    def test_message_key(self):
        assert RESPONSE_KEY_MESSAGE == "message"

    def test_content_key(self):
        assert RESPONSE_KEY_CONTENT == "content"

    def test_response_key(self):
        assert RESPONSE_KEY_RESPONSE == "response"


class TestErrorPrefix:
    def test_error_prefix(self):
        assert ERROR_PREFIX == "Error: "


class TestRoleStrings:
    def test_system(self):
        assert ROLE_SYSTEM == "system"

    def test_user(self):
        assert ROLE_USER == "user"

    def test_assistant(self):
        assert ROLE_ASSISTANT == "assistant"


class TestImageConstants:
    def test_data_url_prefix(self):
        assert IMAGE_DATA_URL_PREFIX == "data:image/jpeg;base64,"

    def test_image_url_type(self):
        assert IMAGE_TYPE_IMAGE_URL == "image_url"

    def test_text_type(self):
        assert IMAGE_TYPE_TEXT == "text"

    def test_image_type(self):
        assert IMAGE_TYPE_IMAGE == "image"

    def test_base64_type(self):
        assert IMAGE_TYPE_BASE64 == "base64"

    def test_media_type_jpeg(self):
        assert MEDIA_TYPE_IMAGE_JPEG == "image/jpeg"


class TestTimeouts:
    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 120

    def test_model_list_timeout(self):
        assert MODEL_LIST_TIMEOUT == 10


class TestGeminiConstants:
    def test_model_role(self):
        assert GEMINI_MODEL_ROLE == "model"


class TestDefaultParameters:
    def test_temperature(self):
        assert DEFAULT_TEMPERATURE == 0.7

    def test_max_tokens(self):
        assert DEFAULT_MAX_TOKENS == 2048

    def test_top_p(self):
        assert DEFAULT_TOP_P == 0.9

    def test_repeat_penalty(self):
        assert DEFAULT_REPEAT_PENALTY == 1.1
