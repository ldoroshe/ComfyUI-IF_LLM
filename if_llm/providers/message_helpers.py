from typing import Any, Dict, List, Optional


def build_base_messages(
    system_message: Optional[str],
    messages: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the base message list shared by most providers.

    Returns a flat list of {"role": ..., "content": ...} dicts.
    System message is prepended if provided. Previous conversation
    messages are included as-is.
    """
    result = []

    if system_message:
        result.append({"role": "system", "content": system_message})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Skip system messages from history (they're handled separately)
        if role == "system":
            continue
        result.append({"role": role, "content": content})

    return result


def build_text_user_message(user_message: str) -> Dict[str, Any]:
    """Build a simple text-only user message."""
    return {"role": "user", "content": user_message}


def build_multimodal_user_message(
    user_message: str,
    base64_images: Optional[List[str]],
    image_format: str = "openai",  # "openai" or "ollama"
) -> Dict[str, Any]:
    """Build a user message with embedded images.

    image_format="openai": uses content array with text + image_url parts
    image_format="ollama": uses content string + images list
    """
    if not base64_images:
        return build_text_user_message(user_message)

    if image_format == "openai":
        content = [{"type": "text", "text": user_message}]
        for img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        return {"role": "user", "content": content}

    elif image_format == "ollama":
        return {
            "role": "user",
            "content": user_message,
            "images": base64_images,
        }

    raise ValueError(f"Unknown image_format: {image_format}")
