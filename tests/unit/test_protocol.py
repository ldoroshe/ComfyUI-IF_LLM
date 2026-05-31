import pytest
from if_llm.providers.base import BaseLLMProvider


class TestNormalizeResponse:
    def test_string_response(self):
        result = BaseLLMProvider.normalize_response("Hello world")
        assert result == {"choices": [{"message": {"content": "Hello world"}}]}

    def test_already_unified_format(self):
        input_data = {"choices": [{"message": {"content": "test"}}]}
        result = BaseLLMProvider.normalize_response(input_data)
        assert result == input_data

    def test_ollama_format_with_response_key(self):
        input_data = {"response": "Ollama answer", "images": ["base64data"]}
        result = BaseLLMProvider.normalize_response(input_data)
        assert result["choices"][0]["message"]["content"] == "Ollama answer"
        assert result["choices"][0]["images"] == ["base64data"]

    def test_ollama_format_with_message_key(self):
        input_data = {"message": {"content": "Message format answer"}}
        result = BaseLLMProvider.normalize_response(input_data)
        assert result["choices"][0]["message"]["content"] == "Message format answer"

    def test_tools_passthrough(self):
        tools_response = {"tool_calls": [...]}
        result = BaseLLMProvider.normalize_response(tools_response, tools=[])
        assert result == tools_response

    def test_fallback_stringify(self):
        result = BaseLLMProvider.normalize_response(12345)
        assert result["choices"][0]["message"]["content"] == "12345"


class TestMakeErrorResponse:
    def test_error_format(self):
        result = BaseLLMProvider.make_error_response("Connection failed")
        assert result == {"choices": [{"message": {"content": "Error: Connection failed"}}]}
