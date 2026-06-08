"""
Shared pytest configuration for comfyui-if_llm tests.

Blocks ComfyUI core imports so tests can load project modules
without a running ComfyUI instance. Provides _load_module helper
to load provider modules via importlib (bypassing relative imports).
"""

import importlib.util
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# CRITICAL: Block ComfyUI core imports before any test file loads
# ---------------------------------------------------------------------------
_COMFYUI_MODULES = {"folder_paths", "node_helpers", "server"}
for _mod in _COMFYUI_MODULES:
    sys.modules[_mod] = MagicMock()


@pytest.fixture(autouse=True)
def _prevent_comfyui_imports(monkeypatch):
    """Ensure no test accidentally imports real ComfyUI."""
    for _mod in _COMFYUI_MODULES:
        monkeypatch.delattr(sys.modules.get(_mod), "base_path", raising=False)


# ---------------------------------------------------------------------------
# Helper: load project modules directly from file (bypasses ComfyUI)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _load_module(name):
    """Load a project module directly from file, avoiding ComfyUI imports.

    Provider modules use relative imports (e.g., 'from .utils import ...')
    that only work when loaded as part of the package. This helper loads
    them via importlib so relative imports resolve within the module file itself.

    Args:
        name: Module filename without .py extension (e.g., 'utils', 'openai_api')

    Returns:
        The loaded module object.
    """
    filepath = os.path.join(_PROJECT_ROOT, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"if_llm_{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures: sample data for tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing message builders."""
    return [
        {"role": "user", "content": "What is in this image?"},
        {"role": "assistant", "content": "This is a photo of a cat."},
    ]


@pytest.fixture
def sample_messages_with_system():
    """Sample conversation messages including a system message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is in this image?"},
        {"role": "assistant", "content": "This is a photo of a cat."},
    ]


@pytest.fixture
def fake_tensor_batch():
    """Small batch of RGB images: [2, 64, 64, 3]."""
    import torch

    return torch.rand(2, 64, 64, 3)


@pytest.fixture
def fake_tensor_single():
    """Single image tensor: [1, 64, 64, 3]."""
    import torch

    return torch.rand(1, 64, 64, 3)


@pytest.fixture
def fake_tensor_chw():
    """Single image tensor in channels-first format: [3, 64, 64]."""
    import torch

    return torch.rand(3, 64, 64)


@pytest.fixture
def fake_tensor_grayscale():
    """Grayscale image tensor: [64, 64]."""
    import torch

    return torch.rand(64, 64)


@pytest.fixture
def sample_base64_images():
    """Generate tiny valid base64 JPEG images for testing."""
    import base64
    import io

    from PIL import Image

    imgs = []
    for idx in range(2):
        color = [(255, 0, 0), (0, 255, 0)][idx]
        img = Image.new("RGB", (10, 10), color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        imgs.append(base64.b64encode(buf.getvalue()).decode())
    return imgs


@pytest.fixture
def sample_simple_image():
    """A minimal 10x10 red PIL image."""
    from PIL import Image

    return Image.new("RGB", (10, 10), color="red")


@pytest.fixture
def sample_png_image():
    """A minimal 10x10 PNG PIL image."""
    import io

    from PIL import Image

    img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image.open(buf)


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """Provide dummy API keys so get_api_key doesn't fail."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-dummy")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-dummy")
    monkeypatch.setenv("GROQ_API_KEY", "groq-dummy")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-dummy")
    monkeypatch.setenv("XAI_API_KEY", "xai-dummy")


@pytest.fixture(autouse=True)
def _suppress_logging(monkeypatch):
    """Suppress logging.basicConfig() calls from provider modules.

    Architecture review notes: multiple files call logging.basicConfig() at module level,
    which pollutes test output. This fixture raises on any attempt to configure logging.
    """
    import logging

    monkeypatch.setattr(logging, "basicConfig", lambda *args, **kwargs: None)


@pytest.fixture
def mock_aioresponse():
    """Provide a mocked aiohttp.ClientSession for async HTTP testing.

    Replaces aioresponses (incompatible with varying aiohttp versions)
    with direct patching of the connection pool's cached session.

    Tests configure mock_response.json() to return their expected payload
    via the yielded mock_response object.
    """

    # Mock response that mimics aiohttp.ClientResponse
    class MockResponse:
        def __init__(self):
            self.status = 200
            self._json = AsyncMock(
                return_value={"choices": [{"message": {"content": "ok"}}]}
            )

        async def json(self):
            return await self._json()

        def raise_for_status(self):
            if self.status >= 400:
                raise Exception(f"HTTP {self.status}")
            return self

    # Async context manager that mimics aiohttp's _BaseRequestContextManager.
    # session.post() returns this object directly (not a coroutine).
    # `async with session.post(...) as resp` enters this context manager and
    # assigns the MockResponse to `resp`.
    class RequestContextManager:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, *args):
            pass

    # Simple session mock that mimics aiohttp.ClientSession.post()
    class MockSession:
        def __init__(self):
            self.closed = False
            self._response = MockResponse()

        def post(self, *args, **kwargs):
            # Return context manager directly (NOT an async function)
            return RequestContextManager(self._response)

    # Patch the connection pool's cached session so all providers use it.
    import if_llm.providers.connection_pool as _pool

    original_session = _pool._session
    mock_session = MockSession()
    _pool._session = mock_session

    yield {
        "session": mock_session,
        "response": mock_session._response,
        "json": mock_session._response._json,
    }

    # Restore original session
    _pool._session = original_session
