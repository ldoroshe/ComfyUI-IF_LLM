#mistral_api.py
import aiohttp
import asyncio
import json
from typing import List, Union, Optional
import requests
import logging
try:
    from mistralai.client.sdk import Mistral
except ImportError:
    from mistralai import Mistral
from if_llm.providers.base import BaseLLMProvider
from if_llm.providers.message_helpers import build_base_messages, build_multimodal_user_message, build_text_user_message
from if_llm.constants import CONTENT_TYPE_JSON, ImageFormat

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def send_mistral_request(base64_images, model, system_message, user_message, messages, api_key, 
                        seed, temperature, max_tokens, top_p, tools=None, tool_choice=None):
    try:
        client = Mistral(api_key=api_key)   

        # Prepare messages using shared helpers
        mistral_messages = prepare_mistral_messages(base64_images, system_message, user_message, messages)

        #logger.debug(f"Sending messages: {json.dumps(mistral_messages, indent=2)}")

        completion = await client.chat.complete_async(
            model=model,
            messages=mistral_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            random_seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            safe_prompt=False
        )

        #logger.debug(f"Received response: {completion}")

        if tools:
            return completion
        else:
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                return {
                    "choices": [{
                        "message": {
                            "content": content
                        }
                    }]
                }
            else:
                error_msg = "Error: No valid choices in the MISTRAL response."
                logger.error(error_msg)
                return BaseLLMProvider.make_error_response(error_msg)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return BaseLLMProvider.make_error_response(f"An unexpected error occurred: {str(e)}")

def prepare_mistral_messages(base64_images, system_message, user_message, messages):
    """Prepare messages for the Mistral API format.
    
    Uses shared helpers from message_helpers module.
    """
    mistral_messages = build_base_messages(system_message, messages)
    
    # Add the current user message with all images if provided
    if base64_images:
        mistral_messages.append(build_multimodal_user_message(user_message, base64_images, image_format=ImageFormat.OPENAI))
        #logger.debug(f"Number of images sent: {len(base64_images)}")
    else:
        mistral_messages.append(build_text_user_message(user_message))
    
    return mistral_messages

async def create_mistral_compatible_embedding(api_key, model, input):
    try:
        client = Mistral(api_key=api_key)
        embedding = await client.embeddings.create(model=model, input=input)
        
        if hasattr(embedding, 'data') and len(embedding.data) > 0 and hasattr(embedding.data[0], 'embedding'):
            return embedding.data[0].embedding  # Return the embedding directly as a list
        elif hasattr(embedding, 'data') and len(embedding.data) == 0:
            raise ValueError("No embedding generated for the input text.")
        else:
            raise ValueError("Unexpected response format: 'embedding' data not found")
    except Exception as e:
        logger.error(f"Error creating Mistral embedding: {str(e)}")
        raise
