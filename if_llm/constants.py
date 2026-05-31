"""Centralized constants for magic strings, numbers, and enums used throughout the codebase."""

from enum import Enum
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Provider type strings — used across providers, send_request, model_utils,
# settings_utils, node_core, ListModelsNode, and every *_api.py file.
# ---------------------------------------------------------------------------

PROVIDER_TYPES: FrozenSet[str] = frozenset([
    "llamacpp", "vllm", "kobold", "openai", "anthropic",
    "gemini", "groq", "mistral", "xai", "deepseek",
    "ollama", "lms", "textgen", "huggingface", "transformers",
])

# Local/offline providers (no API key required)
LOCAL_PROVIDERS: FrozenSet[str] = frozenset([
    "llamacpp", "vllm", "kobold", "ollama", "lms",
    "textgen", "transformers", "huggingface",
])

# Providers that support embeddings
EMBEDDING_PROVIDERS: FrozenSet[str] = frozenset([
    "ollama", "openai", "lmstudio", "llamacpp", "textgen",
    "mistral", "xai",
])


# ---------------------------------------------------------------------------
# Image format strings — used in message_helpers.py and every *_api.py that
# calls build_multimodal_user_message().
# ---------------------------------------------------------------------------

IMAGE_FORMATS: FrozenSet[str] = frozenset(["openai", "ollama", "anthropic", "gemini"])


class ImageFormat(str, Enum):
    """Supported multimodal image formats."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


# ---------------------------------------------------------------------------
# HTTP headers — repeated in 20+ provider files.
# ---------------------------------------------------------------------------

CONTENT_TYPE_JSON = "application/json"


# ---------------------------------------------------------------------------
# Response dictionary keys — used in protocol.py, node_core.py, and every
# provider that normalizes responses.
# ---------------------------------------------------------------------------

RESPONSE_KEY_CHOICES = "choices"
RESPONSE_KEY_MESSAGE = "message"
RESPONSE_KEY_CONTENT = "content"
RESPONSE_KEY_RESPONSE = "response"

ERROR_PREFIX = "Error: "
ERROR_INVALID_IMAGE = "Invalid image data"


# ---------------------------------------------------------------------------
# Message role strings — used in message_helpers.py and every *_api.py.
# ---------------------------------------------------------------------------

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Image data URL prefix — used in image_utils.py, message_helpers.py,
# and lms_api.py.
# ---------------------------------------------------------------------------

IMAGE_DATA_URL_PREFIX = "data:image/jpeg;base64,"


# ---------------------------------------------------------------------------
# Default API timeouts (seconds).
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 120
MODEL_LIST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Image type / media type strings.
# ---------------------------------------------------------------------------

IMAGE_TYPE_IMAGE_URL = "image_url"
IMAGE_TYPE_TEXT = "text"
IMAGE_TYPE_IMAGE = "image"
IMAGE_TYPE_BASE64 = "base64"

MEDIA_TYPE_IMAGE_JPEG = "image/jpeg"


# ---------------------------------------------------------------------------
# Gemini-specific role mapping.
# ---------------------------------------------------------------------------

GEMINI_MODEL_ROLE = "model"


# ---------------------------------------------------------------------------
# Default parameter values (from protocol.py / base.py).
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1
