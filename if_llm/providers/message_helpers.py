from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from if_llm.constants import (
    IMAGE_DATA_URL_PREFIX,
    IMAGE_TYPE_BASE64,
    IMAGE_TYPE_IMAGE,
    IMAGE_TYPE_IMAGE_URL,
    IMAGE_TYPE_TEXT,
    MEDIA_TYPE_IMAGE_JPEG,
    ROLE_USER,
)

# Type alias for a message dict with role and content keys
MessageDict = Dict[str, Any]


def build_base_messages(
    system_message: Optional[str],
    messages: List[MessageDict],
) -> List[MessageDict]:
    """Build the base message list shared by most providers.

    Returns a flat list of {"role": ..., "content": ...} dicts.
    System message is prepended if provided. Previous conversation
    messages are included as-is.

    Args:
        system_message: Optional system prompt to prepend.
        messages: List of conversation message dicts.

    Returns:
        A list of message dicts with role and content keys.
    """
    from if_llm.constants import ROLE_SYSTEM

    result: List[MessageDict] = []

    if system_message:
        result.append({"role": ROLE_SYSTEM, "content": system_message})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Skip system messages from history (they're handled separately)
        if role == ROLE_SYSTEM:
            continue
        result.append({"role": role, "content": content})

    return result


def build_text_user_message(user_message: str) -> MessageDict:
    """Build a simple text-only user message.

    Args:
        user_message: The text content of the user message.

    Returns:
        A dict with role="user" and the given text content.
    """
    return {"role": ROLE_USER, "content": user_message}


def build_multimodal_user_message(
    user_message: str,
    base64_images: Optional[List[str]],
    image_format: Literal["openai", "ollama", "anthropic", "gemini"] = "openai",
) -> MessageDict:
    """Build a user message with embedded images.

    image_format="openai": uses content array with text + image_url parts
    image_format="ollama": uses content string + images list
    image_format="anthropic": uses content array with text + image parts (base64 source)
    image_format="gemini": uses parts array with text + inline_data

    Args:
        user_message: The text content of the user message.
        base64_images: List of base64-encoded image strings.
        image_format: The target API format ("openai", "ollama", "anthropic", "gemini").

    Returns:
        A message dict with embedded images in the specified format.

    Raises:
        ValueError: If image_format is not one of the supported formats.
    """
    if not base64_images:
        return build_text_user_message(user_message)

    if image_format == "openai":
        content: List[Union[Dict[str, str], Dict[str, Any]]] = [
            {"type": IMAGE_TYPE_TEXT, "text": user_message}
        ]
        for img in base64_images:
            content.append(
                {
                    "type": IMAGE_TYPE_IMAGE_URL,
                    "image_url": {"url": f"{IMAGE_DATA_URL_PREFIX}{img}"},
                }
            )
        return {"role": ROLE_USER, "content": content}

    elif image_format == "ollama":
        return {
            "role": ROLE_USER,
            "content": user_message,
            "images": base64_images,
        }

    elif image_format == "anthropic":
        content = [{"type": IMAGE_TYPE_TEXT, "text": user_message}]
        for img in base64_images:
            content.append(
                {
                    "type": IMAGE_TYPE_IMAGE,
                    "source": {
                        "type": IMAGE_TYPE_BASE64,
                        "media_type": MEDIA_TYPE_IMAGE_JPEG,
                        "data": img,
                    },
                }
            )
        return {"role": ROLE_USER, "content": content}

    elif image_format == "gemini":
        parts: List[Union[Dict[str, str], Dict[str, Any]]] = [{"text": user_message}]
        for img in base64_images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": MEDIA_TYPE_IMAGE_JPEG,
                        "data": img,
                    },
                }
            )
        return {"role": ROLE_USER, "parts": parts}

    raise ValueError(f"Unknown image_format: {image_format}")
