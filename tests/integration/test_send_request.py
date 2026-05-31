"""Integration tests for send_request dispatcher routing."""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Mock ComfyUI core before importing send_request
sys.modules['folder_paths'] = MagicMock()
sys.modules['node_helpers'] = MagicMock()
sys.modules['server'] = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()

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

# Load provider modules first so we can reference them
sys.path.insert(0, '..')
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

# Build a namespace with all provider functions for send_request to use
_provider_funcs = {
    'send_anthropic_request': _anthropic_mod.send_anthropic_request,
    'send_ollama_request': _ollama_mod.send_ollama_request,
    'create_ollama_embedding': _ollama_mod.create_ollama_embedding,
    'send_openai_request': _openai_mod.send_openai_request,
    'create_openai_compatible_embedding': _openai_mod.create_openai_compatible_embedding,
    'generate_image': _openai_mod.generate_image,
    'generate_image_variations': _openai_mod.generate_image_variations,
    'edit_image': _openai_mod.edit_image,
    'send_xai_request': _xai_mod.send_xai_request,
    'send_kobold_request': _kobold_mod.send_kobold_request,
    'send_groq_request': _groq_mod.send_groq_request,
    'send_lmstudio_request': _lms_mod.send_lmstudio_request,
    'send_textgen_request': _textgen_mod.send_textgen_request,
    'send_llama_cpp_request': _llamacpp_mod.send_llama_cpp_request,
    'send_mistral_request': _mistral_mod.send_mistral_request,
    'send_vllm_request': _vllm_mod.send_vllm_request,
    'send_gemini_request': _gemini_mod.send_gemini_request,
    'send_deepseek_request': _deepseek_mod.send_deepseek_request,
    'send_huggingface_request': _hf_mod.send_huggingface_request,
    'TransformersModelManager': _transformers_mod.TransformersModelManager,
    'convert_images_for_api': _utils_mod.convert_images_for_api,
    'tensor_to_pil': _utils_mod.tensor_to_pil,
}

# Now load send_request by reading source and replacing relative imports
_spec_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
send_request_path = os.path.join(_spec_root, 'send_request.py')

with open(send_request_path, 'r') as f:
    source = f.read()

# Replace relative imports with comments (we provide the names via _module_ns)
import re
source = re.sub(r'from \.[a-z_]+ import .+', '# relative import replaced by test harness', source)

# Create the module namespace
_module_ns = {'__name__': 'if_llm_send_request', '__file__': send_request_path}
_module_ns.update(_provider_funcs)

# Execute the modified source
exec(compile(source, send_request_path, 'exec'), _module_ns)

send_request = _module_ns['send_request']
format_response = _module_ns['format_response']


class TestSendRequestRouting:
    @pytest.mark.asyncio
    async def test_llamacpp_dispatches_correctly(self, mock_aioresponse):
        mock_aioresponse.post(
            "http://localhost:8000/v1/chat/completions",
            payload={"choices": [{"message": {"content": "llamacpp response"}}]}
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
            payload={"choices": [{"message": {"content": "ollama response"}}]}
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
            payload={"choices": [{"message": {"content": "openai response"}}]}
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
            status=500, payload={"error": "server error"}
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
