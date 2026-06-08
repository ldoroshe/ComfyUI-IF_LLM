from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from if_llm.constants import (
    ERROR_PREFIX,
    RESPONSE_KEY_CHOICES,
    RESPONSE_KEY_CONTENT,
    RESPONSE_KEY_MESSAGE,
    RESPONSE_KEY_RESPONSE,
)


class MessagePart(Protocol):
    """A single part of a multimodal message."""

    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


class LLMResponse(Protocol):
    """Standardized LLM response matching OpenAI format."""

    choices: List[Dict[str, Any]]


class ToolChoice(Protocol):
    """Tool choice specification for function calling."""

    type: str


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
    def normalize_response(
        raw_response: Any,
        tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert provider-specific response to unified format.

        Handles: string -> dict, dict with 'choices' passthrough,
        dict with 'response' key, error responses.

        Args:
            raw_response: The raw response from a provider (string, dict, or other).
            tools: If provided, return the raw response unchanged for tool calling.

        Returns:
            A standardized response dict with 'choices' key.
        """
        if tools is not None:
            return raw_response

        if isinstance(raw_response, str):
            return {
                RESPONSE_KEY_CHOICES: [
                    {RESPONSE_KEY_MESSAGE: {RESPONSE_KEY_CONTENT: raw_response}}
                ]
            }

        if isinstance(raw_response, dict):
            # Already in unified format
            if RESPONSE_KEY_CHOICES in raw_response:
                return raw_response
            # Provider-specific format with 'response' key (e.g., Ollama)
            if RESPONSE_KEY_RESPONSE in raw_response:
                content = raw_response[RESPONSE_KEY_RESPONSE]
                result = {
                    RESPONSE_KEY_CHOICES: [
                        {RESPONSE_KEY_MESSAGE: {RESPONSE_KEY_CONTENT: content}}
                    ]
                }
                if "images" in raw_response:
                    result[RESPONSE_KEY_CHOICES][0]["images"] = raw_response["images"]
                return result
            if RESPONSE_KEY_MESSAGE in raw_response:
                return {
                    RESPONSE_KEY_CHOICES: [
                        {
                            RESPONSE_KEY_MESSAGE: {
                                RESPONSE_KEY_CONTENT: raw_response[
                                    RESPONSE_KEY_MESSAGE
                                ][RESPONSE_KEY_CONTENT]
                            }
                        }
                    ]
                }

        # Fallback: stringify
        return {
            RESPONSE_KEY_CHOICES: [
                {RESPONSE_KEY_MESSAGE: {RESPONSE_KEY_CONTENT: str(raw_response)}}
            ]
        }

    @staticmethod
    def make_error_response(error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response.

        Args:
            error_msg: The error message to include in the response.

        Returns:
            A dict with standardized error format.
        """
        return {
            RESPONSE_KEY_CHOICES: [
                {
                    RESPONSE_KEY_MESSAGE: {
                        RESPONSE_KEY_CONTENT: f"{ERROR_PREFIX}{error_msg}"
                    }
                }
            ]
        }

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

        Args:
            model: The model identifier string.
            system_message: Optional system prompt.
            user_message: The user's input message.
            messages: Previous conversation messages.
            base64_images: Optional list of base64-encoded image strings.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling probability cutoff.
            top_k: Top-k sampling parameter.
            repeat_penalty: Penalty for repeating tokens.
            stop: Optional list of stop sequences.
            seed: Random seed for reproducibility.
            random: If True, use seed; otherwise use temperature.
            tools: Optional tool/function definitions.
            tool_choice: Optional tool choice specification.

        Returns:
            A dict of parameters suitable for passing to a provider API.
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
