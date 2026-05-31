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

# Mock heavy deps only for transformers_api (loaded separately below)
_transformers_mocks = {
    'torch': MagicMock(), 'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(), 'torch.cuda': MagicMock(),
    'transformers': MagicMock(), 'transformers.dynamic_module_utils': MagicMock(),
    'qwen_vl_utils': MagicMock(), 'PIL': MagicMock(),
    'PIL.Image': MagicMock(), 'numpy': MagicMock(),
    'torchvision': MagicMock(), 'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'psutil': MagicMock(), 'huggingface_hub': MagicMock(),
}

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

# Temporarily mock heavy deps only for transformers_api
for _k, _v in _transformers_mocks.items():
    sys.modules[_k] = _v
import transformers_api as _transformers_mod

# Restore heavy deps for other modules (utils.py -> if_llm.image_utils)
for _k in _transformers_mocks:
    sys.modules.pop(_k, None)

import huggingface_api as _hf_mod
import utils as _utils_mod
del _transformers_mocks

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


class TestTransformersRouting:
    @pytest.mark.asyncio
    async def test_transformers_uses_registry(self):
        """Verify transformers provider is in registry and dispatches correctly."""
        # Verify registry contains transformers entry (use _module_ns, not import).
        _PROVIDER_REGISTRY = _module_ns['_PROVIDER_REGISTRY']
        assert "transformers" in _PROVIDER_REGISTRY
        handler, kwargs_builder = _PROVIDER_REGISTRY["transformers"]
        assert callable(handler)
        assert callable(kwargs_builder)

    @pytest.mark.asyncio
    async def test_transformers_dispatches_correctly(self):
        """Verify send_request routes to transformers handler via registry."""
        from unittest.mock import AsyncMock
        # Replace the handler in the registry with a mock.
        _PROVIDER_REGISTRY = _module_ns['_PROVIDER_REGISTRY']
        mock_handler = AsyncMock()
        mock_handler.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        _PROVIDER_REGISTRY['transformers'] = (mock_handler, _module_ns['_build_transformers_kwargs'])

        result = await send_request(
            llm_provider="transformers", base_ip="127.0.0.1", port="8080",
            images=None, llm_model="Qwen/Qwen2.5-7B-Instruct",
            system_message="System prompt", user_message="Hello",
            messages=[], seed=None, temperature=0.7, max_tokens=100,
            random=True, top_k=40, top_p=0.9, repeat_penalty=1.0,
            stop=None, keep_alive=True
        )

        assert result["choices"][0]["message"]["content"] == "test response"
        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert call_kwargs["user_prompt"] == "Hello"
        assert call_kwargs["system_message"] == "System prompt"
        assert call_kwargs["seed"] == 42

    def test_transformers_kwargs_builder(self):
        """Test _build_transformers_kwargs output mapping."""
        _build_transformers_kwargs = _module_ns['_build_transformers_kwargs']

        result = _build_transformers_kwargs(
            base_ip="127.0.0.1", port="8080", formatted_images=None,
            llm_model="Qwen/Qwen2.5-7B-Instruct", system_message="System",
            user_message="Hello", messages=[], seed=None,
            temperature=0.7, max_tokens=100, random=True,
            top_k=40, top_p=0.9, repeat_penalty=1.0,
            stop=["stop"], keep_alive=True, llm_api_key=None,
            tools=None, tool_choice=None, precision="fp16",
            attention="sdpa", aspect_ratio="1:1",
            strategy="normal", mask=None, batch_count=4,
        )

        assert result["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert result["user_prompt"] == "Hello"
        assert result["system_message"] == "System"
        assert result["seed"] == 42
        assert result["random"] is True
        assert result["stop_string"] == "stop"
        assert result["precision"] == "fp16"
        assert result["attention"] == "sdpa"
        assert result["keep_alive"] is True

    def test_transformers_kwargs_builder_defaults(self):
        """Test _build_transformers_kwargs default values."""
        _build_transformers_kwargs = _module_ns['_build_transformers_kwargs']

        result = _build_transformers_kwargs(
            base_ip="127.0.0.1", port="8080", formatted_images=None,
            llm_model="test-model", system_message="", user_message="Hi",
            messages=[], seed=10, temperature=0.5, max_tokens=50,
            random=False, top_k=20, top_p=0.95, repeat_penalty=1.2,
            stop=None, keep_alive=False, llm_api_key=None,
            tools=None, tool_choice=None, precision=None,
            attention=None, aspect_ratio="1:1",
            strategy="normal", mask=None, batch_count=4,
        )

        assert result["seed"] == 10
        assert result["random"] is False
        assert result["stop_string"] == ""
        assert result["precision"] == "fp16"
        assert result["attention"] == "sdpa"


class TestImageConversionErrors:
    @pytest.mark.asyncio
    async def test_image_conversion_failure_returns_error(self):
        """Verify that image conversion failure returns structured error, not None"""
        with patch.dict(_module_ns, {'convert_images_for_api': MagicMock(side_effect=ValueError("Invalid image format"))}):
            result = await send_request(
                llm_provider="ollama", base_ip="127.0.0.1", port="8080",
                images=["invalid-image-data"], llm_model="llava",
                system_message="System", user_message="Hello",
                messages=[], seed=None, temperature=0.7, max_tokens=100,
                random=True, top_k=40, top_p=0.9, repeat_penalty=1.0,
                stop=None, keep_alive=True
            )

            # Should NOT return None
            assert result is not None
            # Should be a dict with error key
            assert isinstance(result, dict)
            assert "error" in result
            assert "Invalid image data" in result["error"]
            # Should have empty choices (consistent with error format)
            assert result.get("choices") == []


class TestSendRequestCompleteFlow:
    @pytest.mark.asyncio
    async def test_complete_transformers_flow(self):
        """Full integration test: transformers provider with valid request."""
        from unittest.mock import AsyncMock
        
        # Replace the handler in the registry with a mock.
        _PROVIDER_REGISTRY = _module_ns['_PROVIDER_REGISTRY']
        mock_handler = AsyncMock()
        mock_handler.return_value = {
            "choices": [{"message": {"content": "Success"}}]
        }
        _PROVIDER_REGISTRY['transformers'] = (mock_handler, _module_ns['_build_transformers_kwargs'])
        
        result = await send_request(
            llm_provider="transformers", base_ip="127.0.0.1", port="8000",
            images=None, llm_model="Qwen/Qwen2.5-7B-Instruct",
            system_message="System", user_message="Hello",
            messages=[], seed=None, temperature=0.7, max_tokens=100,
            random=True, top_k=40, top_p=0.9, repeat_penalty=1.0,
            stop=None, keep_alive=True
        )
        
        assert result is not None
        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Success"

    @pytest.mark.asyncio  
    async def test_invalid_provider_returns_error(self):
        """Verify that invalid provider returns structured error response."""
        result = await send_request(
            llm_provider="invalid_provider", base_ip="127.0.0.1", port="8000",
            images=None, llm_model="test-model",
            system_message="System", user_message="Hello",
            messages=[], seed=None, temperature=0.7, max_tokens=100,
            random=True, top_k=40, top_p=0.9, repeat_penalty=1.0,
            stop=None, keep_alive=True
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert "choices" in result
        assert len(result["choices"]) == 1
        assert "Invalid llm_provider" in result["choices"][0]["message"]["content"]
