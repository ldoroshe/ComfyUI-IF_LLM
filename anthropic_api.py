#anthropic_api.py
import base64
import logging

from anthropic import AsyncAnthropic

from if_llm.providers.base import BaseLLMProvider
from if_llm.providers.message_helpers import build_base_messages

logger = logging.getLogger(__name__)

async def send_anthropic_request(api_key, model, system_message, user_message, messages, temperature, max_tokens, base64_images, tools=None, tool_choice=None):
    try:
        # Create client with minimal parameters
        client = AsyncAnthropic(
            api_key=api_key
        )

        anthropic_messages = prepare_anthropic_messages(user_message, messages, base64_images)

        data = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if system_message:
            data["system"] = system_message

        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        try:
            response = await client.messages.create(**data)

            if tools:
                # If tools were used, return the full response
                return response
            else:
                # If no tools were used, format the response to match the specified structure
                generated_text = response.content[0].text if response.content else ""
                return {
                    "choices": [{
                        "message": {
                            "content": generated_text
                        }
                    }]
                }
        except Exception as e:
            error_msg = f"Error: An exception occurred while processing the Anthropic request: {str(e)}"
            logger.error(error_msg)
            return BaseLLMProvider.make_error_response(error_msg)
    except Exception as e:
        logger.error(f"Error initializing Anthropic client: {str(e)}")
        return BaseLLMProvider.make_error_response(f"Error initializing Anthropic client: {str(e)}")

def detect_image_type(base64_string):
    """
    Detect the image type from a base64 string.
    """
    try:
        # Decode a small portion of the base64 string
        header = base64.b64decode(base64_string[:32])

        # Check for PNG signature
        if header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        # Check for JPEG signature
        elif header.startswith(b'\xff\xd8'):
            return 'image/jpeg'
        # Add more image type checks as needed
        else:
            return 'application/octet-stream'  # Default to binary data
    except Exception:
        return 'application/octet-stream'  # If detection fails, assume binary data

def prepare_anthropic_messages(user_message, messages, base64_images=None):
    """
    Prepares messages for the Anthropic API, ensuring all images are included.

    Uses shared helpers from message_helpers module where applicable.
    Anthropic-specific features (cache_control, user-first ordering) are preserved.

    Args:
        user_message (str): The user's message.
        messages (List[Dict[str, Any]]): Previous messages.
        base64_images (List[str], optional): Base64-encoded images.

    Returns:
        List[Dict[str, Any]]: Formatted messages.
    """
    has_images = base64_images is not None and len(base64_images) > 0

    # Use shared helper to build base messages (skips system from history)
    base_messages = build_base_messages(None, messages)

    # Wrap each message with Anthropic-specific cache_control
    anthropic_messages = []
    for msg in base_messages:
        role = msg["role"]
        content = msg["content"]

        new_message = {"role": role, "content": []}

        if isinstance(content, str):
            new_message["content"].append({"type": "text", "text": content})
        elif isinstance(content, list):
            new_message["content"] = content

        if not has_images:
            if role == "assistant":
                new_message["cache_control"] = {"type": "permanent"}
            elif role == "user":
                new_message["cache_control"] = {"type": "ephemeral"}

        anthropic_messages.append(new_message)

    # Build the current user message with images
    if has_images:
        user_content = [{"type": "text", "text": user_message}]
        for image_data in base64_images:
            media_type = detect_image_type(image_data)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
        new_user_message = {"role": "user", "content": user_content}
    else:
        new_user_message = {"role": "user", "content": [{"type": "text", "text": user_message}]}
        new_user_message["cache_control"] = {"type": "ephemeral"}

    # Anthropic requires conversation to start with user message
    if not anthropic_messages:
        # No history messages — just the new user message
        anthropic_messages.append(new_user_message)
    elif anthropic_messages[0]["role"] == "assistant":
        # History starts with assistant — prepend user message (Anthropic requirement)
        anthropic_messages.insert(0, new_user_message)
        anthropic_messages.append(new_user_message)
    else:
        # History starts with user or we have a system message — append
        anthropic_messages.append(new_user_message)

    return anthropic_messages
