#send_request.py
import asyncio
import logging
from typing import List, Union, Optional, Dict, Any
import base64

# Existing imports
from .anthropic_api import send_anthropic_request
from .ollama_api import send_ollama_request, create_ollama_embedding
from .openai_api import send_openai_request, create_openai_compatible_embedding, generate_image, generate_image_variations, edit_image
from .xai_api import send_xai_request
from .kobold_api import send_kobold_request
from .groq_api import send_groq_request
from .lms_api import send_lmstudio_request
from .textgen_api import send_textgen_request
from .llamacpp_api import send_llama_cpp_request
from .mistral_api import send_mistral_request 
from .vllm_api import send_vllm_request
from .gemini_api import send_gemini_request
from .transformers_api import TransformersModelManager  
from .huggingface_api import send_huggingface_request
from if_llm.constants import ERROR_INVALID_IMAGE
from if_llm.image_utils import convert_images_for_api
from .deepseek_api import send_deepseek_request
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    

# Initialize the TransformersModelManager
_transformers_manager = TransformersModelManager()

# Module-level wrapper for dynamic dispatch via _transformers_manager
async def send_transformers_request(**kwargs):
    return await _transformers_manager.send_transformers_request(**kwargs)


def run_async(coroutine):
    """Helper function to run coroutines in a new event loop if necessary"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    lambda: asyncio.run(coroutine)
                ).result()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)


# =============================================================================
# Provider Registry — replaces the if/elif dispatch chain
# =============================================================================

# --- Kwargs builders: each returns a dict matching its handler's signature ---

def _build_ollama_kwargs(base_ip, port, formatted_images, llm_model, system_message,
                         user_message, messages, seed, temperature, max_tokens,
                         random, top_k, top_p, repeat_penalty, stop, keep_alive,
                         tools, tool_choice, **kwargs):
    api_url = f"http://{base_ip}:{port}/api/chat"
    return {
        "api_url": api_url,
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "random": random,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "keep_alive": keep_alive,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_chat_completions_kwargs(base_ip, port, formatted_images, llm_model,
                                    system_message, user_message, messages, seed,
                                    temperature, max_tokens, top_k, top_p,
                                    repeat_penalty, stop, tools, tool_choice, **kwargs):
    api_url = f"http://{base_ip}:{port}/v1/chat/completions"
    return {
        "api_url": api_url,
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_llamacpp_kwargs(base_ip, port, formatted_images, llm_model,
                            system_message, user_message, messages, seed,
                            temperature, max_tokens, top_k, top_p,
                            repeat_penalty, stop, tools, tool_choice, **kwargs):
    # llamacpp does not support tool_choice
    kwargs = _build_chat_completions_kwargs(
        base_ip, port, formatted_images, llm_model,
        system_message, user_message, messages, seed,
        temperature, max_tokens, top_k, top_p,
        repeat_penalty, stop, tools, tool_choice,
    )
    kwargs.pop("tool_choice", None)
    return kwargs


def _build_vllm_kwargs(base_ip, port, formatted_images, llm_model,
                        system_message, user_message, messages, seed,
                        temperature, max_tokens, top_k, top_p,
                        repeat_penalty, stop, tools, tool_choice,
                        llm_api_key, **kwargs):
    # vllm uses base64_image (singular) and needs api_key
    kwargs = _build_chat_completions_kwargs(
        base_ip, port, formatted_images, llm_model,
        system_message, user_message, messages, seed,
        temperature, max_tokens, top_k, top_p,
        repeat_penalty, stop, tools, tool_choice,
    )
    kwargs["base64_image"] = kwargs.pop("base64_images")
    kwargs["api_key"] = llm_api_key
    return kwargs


def _build_openai_kwargs(formatted_images, llm_model, system_message, user_message,
                         messages, llm_api_key, seed, random, temperature,
                         max_tokens, top_p, repeat_penalty, tools, tool_choice, **kwargs):
    api_url = "https://api.openai.com/v1/chat/completions"
    return {
        "api_url": api_url,
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "api_key": llm_api_key,
        "seed": seed if random else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_xai_kwargs(formatted_images, llm_model, system_message, user_message,
                      messages, llm_api_key, seed, random, temperature,
                      max_tokens, top_p, repeat_penalty, tools, tool_choice, **kwargs):
    api_url = "https://api.x.ai/v1/chat/completions"
    return {
        "api_url": api_url,
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "api_key": llm_api_key,
        "seed": seed if random else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_anthropic_kwargs(llm_api_key, llm_model, system_message, user_message,
                            messages, temperature, max_tokens, formatted_images,
                            tools, tool_choice, **kwargs):
    return {
        "api_key": llm_api_key,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base64_images": formatted_images,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_groq_kwargs(formatted_images, llm_model, system_message, user_message,
                       messages, llm_api_key, temperature, max_tokens, top_p,
                       tools, tool_choice, **kwargs):
    return {
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "api_key": llm_api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_mistral_kwargs(formatted_images, llm_model, system_message, user_message,
                          messages, llm_api_key, seed, random, temperature,
                          max_tokens, top_p, tools, tool_choice, **kwargs):
    return {
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "api_key": llm_api_key,
        "seed": seed if random else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_deepseek_kwargs(formatted_images, llm_model, system_message, user_message,
                           messages, llm_api_key, seed, random, temperature,
                           max_tokens, top_p, tools, tool_choice, **kwargs):
    return {
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "api_key": llm_api_key,
        "seed": seed if random else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_gemini_kwargs(formatted_images, llm_model, system_message, user_message,
                         messages, temperature, max_tokens, top_k, top_p,
                         stop, llm_api_key, tools, tool_choice, **kwargs):
    return {
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "stop": stop,
        "api_key": llm_api_key,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _build_huggingface_kwargs(base_ip, formatted_images, llm_model, system_message,
                              user_message, messages, seed, temperature, max_tokens,
                              top_p, repeat_penalty, stop, keep_alive, llm_api_key,
                              precision, attention, aspect_ratio, strategy, mask,
                              batch_count, **kwargs):
    # Preserves original behaviour: kwargs dict was empty at this point,
    # so .get() calls return defaults / None.
    return {
        "base_ip": base_ip,
        "base64_images": formatted_images,
        "model": llm_model,
        "system_message": system_message,
        "user_message": user_message,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "keep_alive": keep_alive,
        "api_key": llm_api_key,
        "precision": precision,
        "attention": attention,
        "aspect_ratio": aspect_ratio,
        "strategy": strategy,
        "mask": mask,
        "batch_count": batch_count,
    }


def _build_transformers_kwargs(base_ip, port, formatted_images, llm_model,
                                system_message, user_message, messages, seed,
                                temperature, max_tokens, random,
                                top_k, top_p, repeat_penalty, stop, keep_alive,
                                llm_api_key, tools, tool_choice, precision,
                                attention, aspect_ratio, strategy, mask, batch_count, **kwargs):
    """Build kwargs for transformers provider using send_transformers_request signature."""
    return {
        "model_name": llm_model,
        "user_prompt": user_message,
        "system_message": system_message,
        "messages": messages,
        "images": formatted_images,
        "seed": seed if seed is not None else 42,
        "random": random if random else False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop_string": stop[0] if isinstance(stop, list) and len(stop) > 0 else "",
        "precision": precision if precision else "fp16",
        "attention": attention if attention else "sdpa",
        "keep_alive": keep_alive,
        "tools": tools,
        "tool_choice": tool_choice,
    }


# --- Registry: provider alias → (handler_func, kwargs_builder) ---

_PROVIDER_REGISTRY = {
    "ollama":       (send_ollama_request, _build_ollama_kwargs),
    "groq":         (send_groq_request, _build_groq_kwargs),
    "anthropic":    (send_anthropic_request, _build_anthropic_kwargs),
    "openai":       (send_openai_request, _build_openai_kwargs),
    "xai":          (send_xai_request, _build_xai_kwargs),
    "kobold":       (send_kobold_request, _build_chat_completions_kwargs),
    "lmstudio":     (send_lmstudio_request, _build_chat_completions_kwargs),
    "textgen":      (send_textgen_request, _build_chat_completions_kwargs),
    "llamacpp":     (send_llama_cpp_request, _build_llamacpp_kwargs),
    "mistral":      (send_mistral_request, _build_mistral_kwargs),
    "vllm":         (send_vllm_request, _build_vllm_kwargs),
    "gemini":       (send_gemini_request, _build_gemini_kwargs),
    "deepseek":     (send_deepseek_request, _build_deepseek_kwargs),
    "huggingface":  (send_huggingface_request, _build_huggingface_kwargs),
    "transformers": (send_transformers_request, _build_transformers_kwargs),
}


async def send_request(
    llm_provider: str,
    base_ip: str,
    port: str,
    images: List[str],
    llm_model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    seed: Optional[int],
    temperature: float,
    max_tokens: int,
    random: bool,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    stop: Optional[List[str]],
    keep_alive: bool,
    llm_api_key: Optional[str] = None,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    precision: Optional[str] = "fp16", 
    attention: Optional[str] = "sdpa",
    aspect_ratio: Optional[str] = "1:1",
    strategy: Optional[str] = "normal",
    batch_count: Optional[int] = 4,
    mask: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Sends a request to the specified LLM provider and returns a unified response.

    Args:
        llm_provider (str): The LLM provider to use.
        base_ip (str): Base IP address for the API.
        port (int): Port number for the API.
        base64_images (List[str]): List of images encoded in base64.
        llm_model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        seed (Optional[int]): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        random (bool): Whether to use randomness.
        top_k (int): Top K for sampling.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        stop (Optional[List[str]]): Stop sequences.
        keep_alive (bool): Whether to keep the session alive.
        llm_api_key (Optional[str], optional): API key for the LLM provider.
        tools (Optional[Any], optional): Tools to be used.
        tool_choice (Optional[Any], optional): Tool choice.
        precision (Optional[str], optional): Precision for the model.
        attention (Optional[str], optional): Attention mechanism for the model.
        aspect_ratio (Optional[str], optional): Desired aspect ratio for image generation/editing. 
            Options: "1:1", "4:5", "3:4", "5:4", "16:9", "9:16". Defaults to "1:1".
        image_mode (Optional[str], optional): Mode for image processing.
            Options: "create", "edit", "variations". Defaults to "create".

    Returns:
        Union[str, Dict[str, Any]]: Unified response format.
    """
    try:
        # Define aspect ratio to size mapping
        aspect_ratio_mapping = {
            "1:1": "1024x1024",
            "4:5": "1024x1280",
            "3:4": "1024x1365",
            "5:4": "1280x1024",
            "16:9": "1600x900",
            "9:16": "900x1600"
        }

        # Get the size based on the provided aspect_ratio
        size = aspect_ratio_mapping.get(aspect_ratio.lower(), "1024x1024")  # Default to square if invalid

        # Convert images to base64 format for API consumption
        try:
            def is_base64(s):
                try:
                    # Check if string is valid base64
                    return bool(base64.b64encode(base64.b64decode(s)) == s.encode())
                except Exception:
                    return False
            if images is not None and len(images) > 0:
                formatted_images = convert_images_for_api(images, target_format='base64') 
            else:
                formatted_images = None
        except ValueError as ve:
            logger.error(f"Failed to convert images: {str(ve)}", exc_info=True)
            return {
                "error": f"{ERROR_INVALID_IMAGE}: {str(ve)}",
                "choices": [],  # Empty choices on error — consistent with other error paths
            }
        
        #formatted_masks = convert_images_for_api(mask, target_format='base64') if mask is not None and len(mask) > 0 else None

        if llm_provider not in _PROVIDER_REGISTRY:
            raise ValueError(f"Invalid llm_provider: {llm_provider}")

        # --- Registry-based dispatch (replaces the old if/elif chain) ---
        handler, kwargs_builder = _PROVIDER_REGISTRY[llm_provider]

        # Build provider-specific kwargs via the registry builder
        kwargs = kwargs_builder(
            base_ip=base_ip, port=port, formatted_images=formatted_images,
            llm_model=llm_model, system_message=system_message,
            user_message=user_message, messages=messages, seed=seed,
            temperature=temperature, max_tokens=max_tokens, random=random,
            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            stop=stop, keep_alive=keep_alive, llm_api_key=llm_api_key,
            tools=tools, tool_choice=tool_choice, precision=precision,
            attention=attention, aspect_ratio=aspect_ratio,
            strategy=strategy, mask=mask, batch_count=batch_count,
        )

        # OpenAI DALL-E special handling (moved out of kwargs builder)
        if llm_provider == "openai" and llm_model.startswith("dall-e"):
            try:
                # Handle image formatting for edit/variations
                formatted_image = None
                formatted_mask = None

                if images is not None and (strategy == "edit" or strategy == "variations"):
                    # Convert to base64 and take first image only
                    formatted_images = convert_images_for_api(images[0:1], target_format='base64')
                    if formatted_images:
                        formatted_image = formatted_images[0]

                # Handle mask for edit strategy
                if strategy == "edit" and mask is not None:
                    formatted_masks = convert_images_for_api(mask[0:1], target_format='base64')
                    if formatted_masks:
                        formatted_mask = formatted_masks[0]

                # Make appropriate API call based on strategy
                if strategy == "create":
                    response = await generate_image(
                        prompt=user_message,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key
                    )
                elif strategy == "edit":
                    response = await edit_image(
                        image_base64=formatted_image,
                        mask_base64=formatted_mask,
                        prompt=user_message,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key
                    )
                elif strategy == "variations":
                    response = await generate_image_variations(
                        image_base64=formatted_image,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key
                    )
                else:
                    raise ValueError(f"Invalid strategy: {strategy}")

                # Return the response directly - it will be a list of base64 strings
                return {"images": response}

            except Exception as e:
                error_msg = f"Error in DALL·E {strategy}: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

        # Call the handler with built kwargs
        response = await handler(**kwargs)

        # Ensure response is properly awaited if it's a coroutine
        if asyncio.iscoroutine(response):
            response = await response

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "content" in choices[0].get("message", {}):
                content = choices[0]["message"]["content"]
                if content.startswith("Error:"):
                    logger.error(f"Error from {llm_provider} API: {content}")

        if tools:
            return response
        try:
            if isinstance(response, dict) and "choices" in response:
                return response
            elif isinstance(response, str):
                return {
                    "choices": [{
                        "message": {
                            "content": response
                        }
                    }]
                }
            else:
                error_msg = f"Unexpected response format: {type(response)}"
                logger.error(error_msg)
                return {
                    "choices": [{
                        "message": {
                            "content": error_msg
                        }
                    }]
                }
        except Exception as e:
            error_msg = f"Error formatting response: {str(e)}"
            logger.error(error_msg)
            return {
                "choices": [{
                    "message": {
                        "content": error_msg
                    }
                }]
            }

    except Exception as e:
        logger.error(f"Exception in send_request: {str(e)}", exc_info=True)
        return {"choices": [{"message": {"content": f"Exception: {str(e)}"}}]}

def format_response(response, tools):
    """Helper function to format the response consistently"""
    if tools:
        return response
    try:
        if isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if not choices:
                return f"Empty response from provider (no choices returned)"
            return choices[0]["message"]["content"]
        return response
    except (KeyError, IndexError, TypeError) as e:
        error_msg = f"Error formatting response: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def create_embedding(embedding_provider: str, api_base: str, embedding_model: str, input: Union[str, List[str]], embedding_api_key: Optional[str] = None) -> Union[List[float], None]: # Correct return type hint
    if embedding_provider == "ollama":
        return await create_ollama_embedding(api_base, embedding_model, input)
    
    
    elif embedding_provider in ["openai", "lmstudio", "llamacpp", "textgen", "mistral", "xai"]:
        try:
            return await create_openai_compatible_embedding(api_base, embedding_model, input, embedding_api_key) # Try block for more precise error handling
        except ValueError as e:
            logger.error(f"Error creating embedding: {e}")  
            return None # Return None on error
    
    else:
        raise ValueError(f"Unsupported embedding_provider: {embedding_provider}")
