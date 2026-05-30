"""Tests for provider message builder functions."""

import sys
import io
from unittest.mock import MagicMock

# Mock external API libraries BEFORE loading provider modules
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

# Load provider modules directly from file
sys.path.insert(0, '..')
import deepseek_api as _deepseek_api
import llamacpp_api as _llamacpp_api
import mistral_api as _mistral_api
import anthropic_api as _anthropic_api
import openai_api as _openai_api
import gemini_api as _gemini_api
import groq_api as _groq_api
import ollama_api as _ollama_api
import kobold_api as _kobold_api
import lms_api as _lms_api
import textgen_api as _textgen_api
import vllm_api as _vllm_api
import xai_api as _xai_api


class TestPrepareDeepseekMessages:
    def test_basic_user_message(self):
        result = _deepseek_api.prepare_deepseek_messages(
            system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_with_system_message(self):
        result = _deepseek_api.prepare_deepseek_messages(
            system_message="You are helpful.", user_message="hello", messages=[]
        )
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}

    def test_with_conversation_history(self):
        messages = [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "This is a photo of a cat."},
        ]
        result = _deepseek_api.prepare_deepseek_messages(
            system_message="", user_message="follow up", messages=messages
        )
        assert len(result) == 3

    def test_filters_non_string_content(self):
        messages = [
            {"role": "user", "content": "valid text"},
            {"role": "assistant", "content": ["image_block"]},
        ]
        result = _deepseek_api.prepare_deepseek_messages(
            system_message="", user_message="hi", messages=messages
        )
        assert len(result) == 2


class TestPrepareLlamaCppMessages:
    def test_basic_user_message(self):
        result = _llamacpp_api.prepare_llama_cpp_messages(
            system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1

    def test_with_system_message(self):
        result = _llamacpp_api.prepare_llama_cpp_messages(
            system_message="be helpful", user_message="hello", messages=[]
        )
        assert result[0] == {"role": "system", "content": "be helpful"}

    def test_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _llamacpp_api.prepare_llama_cpp_messages(
            system_message="", user_message="describe", messages=[],
            base64_images=imgs
        )
        assert len(result) == 1
        user_msg = result[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "describe"}
        assert user_msg["content"][1]["type"] == "image_url"


class TestPrepareMistralMessages:
    def test_basic_user_message(self):
        result = _mistral_api.prepare_mistral_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1

    def test_with_system_message(self):
        result = _mistral_api.prepare_mistral_messages(
            base64_images=None, system_message="be helpful", user_message="hello", messages=[]
        )
        assert result[0] == {"role": "system", "content": "be helpful"}

    def test_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _mistral_api.prepare_mistral_messages(
            base64_images=imgs, system_message="", user_message="describe", messages=[]
        )
        assert len(result) == 1
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"


class TestPrepareAnthropicMessages:
    def test_basic_user_message(self):
        result = _anthropic_api.prepare_anthropic_messages(
            user_message="hello", messages=[], base64_images=None
        )
        assert len(result) >= 1

    def test_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _anthropic_api.prepare_anthropic_messages(
            user_message="describe", messages=[], base64_images=imgs
        )
        assert len(result) >= 1
        user_content = result[0]["content"]
        image_blocks = [c for c in user_content if c.get("type") == "image"]
        assert len(image_blocks) > 0

    def test_skips_system_in_history(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "This is a photo of a cat."},
        ]
        result = _anthropic_api.prepare_anthropic_messages(
            user_message="next", messages=messages, base64_images=None
        )
        roles = [m["role"] for m in result]
        assert roles[0] == "user"


class TestPrepareOpenaiMessages:
    def test_basic_user_message(self):
        result = _openai_api.prepare_openai_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1

    def test_with_system_message(self):
        result = _openai_api.prepare_openai_messages(
            base64_images=None, system_message="be helpful", user_message="hello", messages=[]
        )
        assert result[0] == {"role": "system", "content": "be helpful"}

    def test_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _openai_api.prepare_openai_messages(
            base64_images=imgs, system_message="", user_message="describe", messages=[]
        )
        assert len(result) == 1
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"


class TestPrepareGeminiMessages:
    def test_basic_user_message(self):
        result = _gemini_api.prepare_gemini_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) >= 1


class TestPrepareGroqMessages:
    def test_basic_user_message(self):
        result = _groq_api.prepare_groq_messages(
            base64_images=[], system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1

    def test_system_omitted_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _groq_api.prepare_groq_messages(
            base64_images=imgs, system_message="be helpful", user_message="describe", messages=[]
        )
        roles = [m.get("role") for m in result]
        assert "system" not in roles


class TestPrepareOllamaMessages:
    def test_basic_user_message(self):
        result = _ollama_api.prepare_ollama_messages(
            system_message="", user_message="hello", messages=[], base64_images=None
        )
        assert len(result) == 2

    def test_with_system_message(self):
        result = _ollama_api.prepare_ollama_messages(
            system_message="be helpful", user_message="hello", messages=[]
        )
        assert result[0] == {"role": "system", "content": "be helpful"}


class TestPrepareKoboldMessages:
    def test_basic_user_message(self):
        result = _kobold_api.prepare_kobold_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1


class TestPrepareLmsMessages:
    def test_basic_user_message(self):
        result = _lms_api.prepare_lmstudio_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1


class TestPrepareTextgenMessages:
    def test_basic_user_message(self):
        result = _textgen_api.prepare_textgen_messages(
            system_message="", user_message="hello", messages=[], base64_image=None
        )
        assert len(result) == 1


class TestPrepareVllmMessages:
    def test_basic_user_message(self):
        result = _vllm_api.prepare_vllm_messages(
            system_message="", user_message="hello", messages=[], base64_image=None
        )
        assert len(result) == 2


class TestPrepareXaiMessages:
    def test_basic_user_message(self):
        result = _xai_api.prepare_xai_messages(
            base64_images=None, system_message="", user_message="hello", messages=[]
        )
        assert len(result) == 1

    def test_with_images(self):
        from PIL import Image
        import base64
        imgs = []
        for _ in range(2):
            img = Image.new("RGB", (10, 10), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        result = _xai_api.prepare_xai_messages(
            base64_images=imgs, system_message="", user_message="describe", messages=[]
        )
        assert len(result) == 1
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
