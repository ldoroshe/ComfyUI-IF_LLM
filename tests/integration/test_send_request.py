"""Integration tests for send_request dispatcher routing."""

import sys
from unittest.mock import MagicMock, patch

# Mock ComfyUI core before importing send_request
sys.modules['folder_paths'] = MagicMock()

# Mock external API libraries that provider modules import at load time
sys.modules['anthropic'] = MagicMock()
sys.modules['anthropic.types'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['groq'] = MagicMock()
sys.modules['mistralai'] = MagicMock()
sys.modules['mistralai.client'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['openai.types'] = MagicMock()
sys.modules['openai.types.chat'] = MagicMock()
sys.modules['xai'] = MagicMock()

# Load provider modules first so relative imports in send_request can resolve
sys.path.insert(0, '..')

# Pre-register provider modules so the relative imports in send_request work
import anthropic_api as _anthropic_mod
import ollama_api as _ollama_mod
import openai_api as _openai_mod
import xai_api as _xai_mod
import kobold_api as _kobold_mod
import groq_api as _groq_mod
import lms_api as _lms_mod
import textgen_api as _textgen_mod
import llamacpp_api as _llamacpp_mod
import mistral_api as _mistral_mod
import vllm_api as _vllm_mod
import gemini_api as _gemini_mod
import deepseek_api as _deepseek_mod
import transformers_api as _transformers_mod
import huggingface_api as _hf_mod
import utils as _utils_mod

# Register them under package-qualified names so relative imports work
sys.modules['if_llm.anthropic_api'] = _anthropic_mod
sys.modules['if_llm.ollama_api'] = _ollama_mod
sys.modules['if_llm.openai_api'] = _openai_mod
sys.modules['if_llm.xai_api'] = _xai_mod
sys.modules['if_llm.kobold_api'] = _kobold_mod
sys.modules['if_llm.groq_api'] = _groq_mod
sys.modules['if_llm.lms_api'] = _lms_mod
sys.modules['if_llm.textgen_api'] = _textgen_mod
sys.modules['if_llm.llamacpp_api'] = _llamacpp_mod
sys.modules['if_llm.mistral_api'] = _mistral_mod
sys.modules['if_llm.vllm_api'] = _vllm_mod
sys.modules['if_llm.gemini_api'] = _gemini_mod
sys.modules['if_llm.deepseek_api'] = _deepseek_mod
sys.modules['if_llm.transformers_api'] = _transformers_mod
sys.modules['if_llm.huggingface_api'] = _hf_mod
sys.modules['if_llm.utils'] = _utils_mod

# Now load send_request via importlib (relative imports will resolve to our mocks)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'if_llm_send_request',
    '../send_request.py'
)
send_request_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(send_request_mod)

send_request = send_request_mod.send_request
format_response = send_request_mod.format_response


class TestSendRequestRouting:
    @pytest.mark.asyncio
    async def test_llamacpp_dispatches_correctly(self, mock_aioresponse):
        mock_aioresponse.post(
            "http://localhost:8000/v1/chat/completions",
            json={"choices": [{"message": {"content": "llamacpp response"}}]}
        )

        result = await send_request(
            llm_provider="llamacpp", base_ip="localhost", port="8000",
            images=None, llm_model="test-model", system_message="",
            user_message="hello", messages=[], seed=None, temperature=0.8,
            max_tokens=100, random=False, top_k=40, top_p=0.9,
            repeat_penalty=1.1, stop=None, keep_alive=False
        )

        assert result["choices"][0]["message"]["content"] == "llamacpp response"

    @pytest.mark.asyncio
    async def test_ollama_dispatches_correctly(self, mock_aioresponse):
        mock_aioresponse.post(
            "http://localhost:11434/api/chat",
            json={"choices": [{"message": {"content": "ollama response"}}]}
        )

        result = await send_request(
            llm_provider="ollama", base_ip="localhost", port="11434",
            images=None, llm_model="llama3", system_message="",
            user_message="hello", messages=[], seed=None, temperature=0.8,
            max_tokens=100, random=False, top_k=40, top_p=0.9,
            repeat_penalty=1.1, stop=None, keep_alive=False
        )

        assert result["choices"][0]["message"]["content"] == "ollama response"

    @pytest.mark.asyncio
    async def test_openai_dispatches_correctly(self, mock_aioresponse):
        mock_aioresponse.post(
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "openai response"}}]}
        )

        result = await send_request(
            llm_provider="openai", base_ip="", port="",
            images=None, llm_model="gpt-4", system_message="",
            user_message="hello", messages=[], seed=None, temperature=0.8,
            max_tokens=100, random=False, top_k=40, top_p=0.9,
            repeat_penalty=1.1, stop=None, keep_alive=False,
            llm_api_key="sk-dummy"
        )

        assert result["choices"][0]["message"]["content"] == "openai response"


class TestSendRequestErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error_returns_error_format(self, mock_aioresponse):
        mock_aioresponse.post(
            "http://localhost:8000/v1/chat/completions",
            status=500, json={"error": "server error"}
        )

        result = await send_request(
            llm_provider="llamacpp", base_ip="localhost", port="8000",
            images=None, llm_model="test-model", system_message="",
            user_message="hello", messages=[], seed=None, temperature=0.8,
            max_tokens=100, random=False, top_k=40, top_p=0.9,
            repeat_penalty=1.1, stop=None, keep_alive=False
        )

        assert "choices" in result


class TestFormatResponse:
    def test_simple_text_response(self):
        response = {"choices": [{"message": {"content": "hello world"}}]}
        result = format_response(response, tools=None)
        assert result == "hello world"

    def test_empty_choices(self):
        response = {"choices": []}
        result = format_response(response, tools=None)
        assert isinstance(result, str)

    def test_tools_enabled_returns_raw(self):
        response = {"choices": [{"message": {"content": "tool call"}}]}
        result = format_response(response, tools=True)
        assert result == response
