from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Protocol


class MessagePart(Protocol):
    """A single part of a multimodal message."""
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


class LLMResponse(Protocol):
    """Standardized LLM response matching OpenAI format."""
    choices: List[Dict[str, Any]]


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Every provider MUST:
    - Implement send_chat() as async
    - Return responses in unified format: {"choices": [{"message": {"content": str}}]}
    - Handle its own authentication and URL construction
    """

    @abstractmethod
    async def send_chat(
        self,
        model: str,
        system_message: Optional[str],
        user_message: str,
        messages: List[Dict[str, Any]],
        base64_images: Optional[List[str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        random: bool = False,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        keep_alive: bool = False,
    ) -> Dict[str, Any]:
        """Send a chat request and return standardized response.

        All providers MUST implement this as async.
        Return format: {"choices": [{"message": {"content": str}}]}
        """
        ...

    @staticmethod
    def normalize_response(raw_response: Any, tools: Optional[Any] = None) -> Dict[str, Any]:
        """Convert provider-specific response to unified format.

        Handles: string -> dict, dict with 'choices' passthrough,
        dict with 'response' key, error responses.
        """
        if tools is not None:
            return raw_response

        if isinstance(raw_response, str):
            return {"choices": [{"message": {"content": raw_response}}]}

        if isinstance(raw_response, dict):
            # Already in unified format
            if "choices" in raw_response:
                return raw_response
            # Provider-specific format with 'response' key (e.g., Ollama)
            if "response" in raw_response:
                content = raw_response["response"]
                result = {"choices": [{"message": {"content": content}}]}
                if "images" in raw_response:
                    result["choices"][0]["images"] = raw_response["images"]
                return result
            if "message" in raw_response:
                return {"choices": [{"message": {"content": raw_response["message"]["content"]}}]}

        # Fallback: stringify
        return {"choices": [{"message": {"content": str(raw_response)}}]}

    @staticmethod
    def make_error_response(error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {"choices": [{"message": {"content": f"Error: {error_msg}"}}]}

    @staticmethod
    def build_common_kwargs(
        model: str,
        system_message: Optional[str],
        user_message: str,
        messages: List[Dict[str, Any]],
        base64_images: Optional[List[str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        random: bool = False,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build common kwargs dict shared by most provider builders.

        Subclasses can call this and then add/remove provider-specific keys.
        """
        kwargs = {
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "base64_images": base64_images,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if seed is not None and random:
            kwargs["seed"] = seed
        else:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k
        return kwargs
