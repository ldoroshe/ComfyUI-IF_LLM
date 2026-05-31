"""Tests for shared message preparation helper functions."""

import pytest
from if_llm.providers.message_helpers import (
    build_base_messages, build_text_user_message, build_multimodal_user_message,
)


class TestBuildBaseMessages:
    def test_empty(self):
        assert build_base_messages(None, []) == []

    def test_with_system(self):
        result = build_base_messages("You are helpful.", [])
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = build_base_messages(None, history)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_skips_system_in_history(self):
        history = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "Hi"},
        ]
        result = build_base_messages(None, history)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestBuildTextUserMessage:
    def test_simple(self):
        result = build_text_user_message("Hello")
        assert result == {"role": "user", "content": "Hello"}


class TestBuildMultimodalUserMessage:
    def test_no_images(self):
        result = build_multimodal_user_message("Describe this", None)
        assert result == {"role": "user", "content": "Describe this"}

    def test_openai_format(self):
        result = build_multimodal_user_message(
            "Describe this", ["abc123"], image_format="openai"
        )
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image_url"

    def test_ollama_format(self):
        result = build_multimodal_user_message(
            "Describe this", ["abc123"], image_format="ollama"
        )
        assert result == {
            "role": "user",
            "content": "Describe this",
            "images": ["abc123"],
        }

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unknown image_format"):
            build_multimodal_user_message("test", ["img"], image_format="invalid")
