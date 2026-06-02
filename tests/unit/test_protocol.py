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


class TestBuildCommonKwargs:
    def test_basic_params(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message="You are helpful.",
            user_message="Hello",
            messages=[],
            base64_images=None,
        )
        assert result["model"] == "gpt-4"
        assert result["system_message"] == "You are helpful."
        assert result["user_message"] == "Hello"
        assert result["messages"] == []
        assert result["base64_images"] is None

    def test_default_values(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=None,
        )
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 2048
        assert result["top_p"] == 0.9
        assert result["repeat_penalty"] == 1.1

    def test_seed_with_random_true(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=None,
            seed=42,
            random=True,
        )
        assert result["seed"] == 42

    def test_seed_with_random_false(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=None,
            seed=42,
            random=False,
        )
        assert "seed" not in result

    def test_top_k_included(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=None,
            top_k=50,
        )
        assert result["top_k"] == 50

    def test_top_k_not_included_when_none(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=None,
        )
        assert "top_k" not in result

    def test_optional_params(self):
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=[],
            base64_images=["img1"],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.95,
            repeat_penalty=1.2,
            stop=["\n"],
            tools=[{"type": "function"}],
            tool_choice={"type": "auto"},
        )
        assert result["base64_images"] == ["img1"]
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 1024
        assert result["top_p"] == 0.95
        assert result["repeat_penalty"] == 1.2
        assert result["stop"] == ["\n"]
        assert result["tools"] == [{"type": "function"}]
        assert result["tool_choice"] == {"type": "auto"}

    def test_does_not_mutate_inputs(self):
        messages = [{"role": "user", "content": "Hi"}]
        images = ["abc"]
        result = BaseLLMProvider.build_common_kwargs(
            model="gpt-4",
            system_message=None,
            user_message="Hello",
            messages=messages,
            base64_images=images,
        )
        assert result["messages"] is messages
        assert result["base64_images"] is images
