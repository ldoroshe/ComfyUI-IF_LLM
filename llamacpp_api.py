#llamacpp_api.py
import requests
import json
import base64
import aiohttp
import logging
from typing import List, Union, Optional, Dict, Any
import base64
import os
from if_llm.providers.base import BaseLLMProvider
from if_llm.providers.message_helpers import build_base_messages, build_multimodal_user_message, build_text_user_message
from if_llm.providers.connection_pool import get_session

logger = logging.getLogger(__name__)

async def send_llama_cpp_request(api_url, base64_images, model, system_message, user_message, messages, seed, 
                           temperature, max_tokens, top_k, top_p, repeat_penalty, stop, tools=None):
    headers = {
        "Content-Type": "application/json"
    }

    #api_url = f"{api_url}/v1/chat/completions"
    
    data = {
        "model": model,
        "messages": prepare_llama_cpp_messages(system_message, user_message, messages, base64_images),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "frequency_penalty": repeat_penalty,
        "stop": stop,
        "seed": seed
    }
    

    try:
        session = await get_session()
        async with session.post(api_url, headers=headers, json=data) as response:
            response.raise_for_status()
            response_data = await response.json()
            return BaseLLMProvider.normalize_response(response_data, tools=tools)

    except Exception as e:
        logger.error(f"Error in LLaMa.cpp API request: {e}")
        return BaseLLMProvider.make_error_response(str(e))

def prepare_llama_cpp_messages(system_message, user_message, messages, base64_images=None):
    llama_messages = build_base_messages(system_message, messages)

    if base64_images:
        llama_messages.append(build_multimodal_user_message(user_message, base64_images, image_format="openai"))
    else:
        llama_messages.append(build_text_user_message(user_message))
    
    return llama_messages
