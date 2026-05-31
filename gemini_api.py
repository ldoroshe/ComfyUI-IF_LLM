# gemini_api.py
import aiohttp
import json
import logging
import asyncio
from if_llm.providers.base import BaseLLMProvider
from if_llm.providers.message_helpers import build_base_messages, build_multimodal_user_message
from if_llm.providers.connection_pool import get_session

logger = logging.getLogger(__name__)

async def send_gemini_request(base64_images, model, system_message, user_message, messages,
                             temperature, max_tokens, top_k, top_p, stop, api_key,
                             tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }
    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    # Append the API key to the URL
    api_url = f"{base_url}?key={api_key}"
    
    gemini_messages = prepare_gemini_messages(base64_images, system_message, user_message, messages)
    
    data = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p, 
            "topK": top_k,
            "maxOutputTokens": max_tokens,
            "stopSequences": stop if isinstance(stop, list) else [stop]
        }
    }

    if tools:
        data["generationConfig"]["tools"] = [{"functionDeclarations": tools}] # Changed to functionDeclarations

    if tool_choice:
        data["toolChoice"] = tool_choice  # Assuming Gemini supports this

    try:
        session = await get_session()
        async with session.post(api_url, headers=headers, json=data) as response:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = await response.json()

            if tools:
                return response_data
            else:
                return BaseLLMProvider.normalize_response(response_data, tools=tools)

    except Exception as e:
        error_msg = "Unexpected error during Gemini API call"
        # Log the full error for debugging but return sanitized message
        logger.error(f"{error_msg}: {str(e)}")
        return BaseLLMProvider.make_error_response(error_msg)

def prepare_gemini_messages(base64_images, system_message, user_message, messages):
    """Prepare messages for the Gemini API format.
    
    Uses shared helpers from message_helpers module where applicable.
    Gemini-specific features (role mapping, system as user prefix) are preserved.
    """
    gemini_messages = []

    # Add system message if provided (Gemini uses user role with "System:" prefix)
    if system_message:
        gemini_messages.append({"role": "user", "parts": [{"text": f"System: {system_message}"}]})

    # Use shared helper for history (skips system messages)
    base_messages = build_base_messages(None, messages)

    # Transform history for Gemini format (assistant -> model)
    for message in base_messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        content = message["content"]
        
        if isinstance(content, list):
            gemini_messages.append({"role": role, "parts": content})
        else:
            gemini_messages.append({"role": role, "parts": [{"text": content}]})

    # Add current user message with images
    if base64_images:
        gemini_messages.append(build_multimodal_user_message(user_message, base64_images, image_format="gemini"))
    else:
        gemini_messages.append({"role": "user", "parts": [{"text": user_message}]})
    
    return gemini_messages
