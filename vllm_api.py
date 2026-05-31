import aiohttp
import json

from if_llm.providers.base import BaseLLMProvider
from if_llm.providers.message_helpers import build_base_messages, build_multimodal_user_message, build_text_user_message
from if_llm.providers.connection_pool import get_session


def prepare_vllm_messages(system_message, user_message, messages, base64_image=None):
    """Build vllm messages — always includes system message entry for compatibility."""
    vllm_messages = build_base_messages(system_message, messages)
    # vllm always includes a system message entry for compatibility
    if not vllm_messages or vllm_messages[0].get("role") != "system":
        vllm_messages.insert(0, {"role": "system", "content": system_message or ""})

    if base64_image:
        vllm_messages.append(build_multimodal_user_message(user_message, [base64_image], image_format="openai"))
    else:
        vllm_messages.append(build_text_user_message(user_message))

    return vllm_messages


async def send_vllm_request(api_url, base64_image, model, system_message, user_message, messages, seed, 
                            temperature, max_tokens, top_k, top_p, repeat_penalty, stop, api_key,
                            tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": prepare_vllm_messages(system_message, user_message, messages, base64_image),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repeat_penalty,
        "stop": stop,
        "seed": seed
    }

    if tools:
        data["functions"] = tools
    if tool_choice:
        if tool_choice == "auto":
            data["function_call"] = "auto"
        elif tool_choice == "none":
            data["function_call"] = "none"
        else:
            data["function_call"] = {"name": tool_choice["function"]["name"]}

    try:
        session = await get_session()
        async with session.post(api_url, headers=headers, data=json.dumps(data)) as response:
            response.raise_for_status()
            response_data = await response.json()
    except Exception as e:
        raise BaseLLMProvider.make_error_response(str(e))["choices"][0]["message"]["content"]

    message = response_data["choices"][0]["message"]
    
    if "function_call" in message:
        return {
            "function_call": {
                "name": message["function_call"]["name"],
                "arguments": message["function_call"]["arguments"]
            }
        }, messages
    else:
        return message["content"], messages
