"""Model listing and API key management utilities."""
import os
import logging

import requests
from dotenv import load_dotenv

from if_llm.constants import CONTENT_TYPE_JSON, MODEL_LIST_TIMEOUT

logger = logging.getLogger(__name__)


def get_api_key(api_key_name, engine):
    """
    Retrieve API key from environment variables or .env file.

    Args:
        api_key_name (str): Name of the API key environment variable
        engine (str): Name of the engine being used

    Returns:
        str: API key if found and valid

    Raises:
        ValueError: If API key is missing or invalid
    """
    local_engines = ["ollama", "llamacpp", "kobold", "lmstudio", "textgen",
                     "sentence_transformers", "transformers"]
    # Try to get the key from .env first
    load_dotenv()
    api_key = os.getenv(api_key_name)
    if engine.lower() in local_engines:
        logger.debug(f"You are using {engine} as the engine, no API key is required.")
        return "1234"

    # Special handling for HuggingFace
    if engine.lower() == "huggingface":
        # Try both conventional and HF-specific env var names
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_AUTH_TOKEN")
        if api_key:
            if validate_huggingface_token(api_key):
                return api_key
            else:
                raise ValueError("Invalid HuggingFace API key")
        raise ValueError("No HuggingFace API key found in environment variables")

    elif api_key:
        logger.debug(f"API key for {api_key_name} found in .env file or environment variables")
        return api_key

    logger.error(f"API key for {api_key_name} not found in .env file or environment variables")
    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")


def validate_huggingface_token(api_key):
    """Validate HuggingFace API token"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        # Try to access the API with the token
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error validating HuggingFace token: {e}")
        return False


def get_models(engine, base_ip, port, api_key):

    if engine == "ollama":
        api_url = f"http://{base_ip}:{port}/api/tags"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from Ollama: {e}")
            return []

    elif engine == "huggingface":
        fallback_models = [
            # Vision Language Models (VLM)
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "Qwen/Qwen2-VL-7B-Chat",
            "Qwen/Qwen2-VL-7B",
            "Qwen/Qwen2-VL-2B-Chat",
            "Qwen/Qwen2-VL-2B",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "microsoft/phi-2",
            "HuggingFaceH4/zephyr-7b-beta",

            # Text to Image Models
            "stabilityai/sdxl-turbo",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-3-base",
            "stabilityai/stable-diffusion-3-medium",
            "stabilityai/stable-diffusion-3-small",
            "black-forest-labs/FLUX.1-dev",
            "playgroundai/playground-v2-256px",
            "playgroundai/playground-v2-1024px",

            # Image to Image Models
            "timbrooks/instruct-pix2pix",
            "lambdalabs/sd-image-variations-diffusers",
            "diffusers/controlnet-canny-sdxl-1.0",

            # Specialized Models
            "kandinsky-community/kandinsky-3",
            "stabilityai/stable-cascade",
            "dataautogpt3/OpenDalle3",
            "ByteDance/SDXL-Lightning",

            # ControlNet Models
            "lllyasviel/control_v11p_sd15_canny",
            "lllyasviel/control_v11p_sd15_openpose",
            "lllyasviel/control_v11p_sd15_depth",

            # Text Feature Extraction
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",

            # Image Feature Extraction
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",

            # Text Classification
            "distilbert-base-uncased-finetuned-sst-2-english",
            "roberta-base-openai-detector",

            # Text Generation
            "gpt2",
            "facebook/opt-350m",

            # Translation
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-fr-en",

            # Question Answering
            "deepset/roberta-base-squad2",
            "distilbert-base-cased-distilled-squad"
        ]

        try:
            # Verify API key
            if not api_key or api_key == "1234":
                logger.warning("No valid HuggingFace API key provided. Using fallback models.")
                return fallback_models

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": CONTENT_TYPE_JSON
            }

            # Check inference API endpoint directly
            inference_url = "https://api-inference.huggingface.co/status"
            response = requests.get(inference_url, headers=headers)

            if response.status_code != 200:
                logger.warning("Failed to verify HuggingFace Inference API access. Using fallback models.")
                return fallback_models

            # Get models available for inference API
            models_url = "https://api-inference.huggingface.co/framework/all"
            response = requests.get(models_url, headers=headers)

            if response.status_code == 200:
                api_models = []
                data = response.json()

                # Extract models supporting inference API
                for framework in data:
                    for model in framework.get("models", []):
                        model_id = model.get("model_id")
                        if model_id:
                            api_models.append(model_id)

                # Combine with fallback models and remove duplicates
                combined_models = list(dict.fromkeys(api_models + fallback_models))
                return combined_models
            else:
                logger.warning(f"Failed to fetch inference models. Status code: {response.status_code}")
                return fallback_models

        except Exception as e:
            logger.error(f"Error fetching HuggingFace models: {str(e)}")
            return fallback_models

    elif engine == "deepseek":
        fallback_models = [
            "deepseek-reasoner",
            "deepseek-chat",
            "deepseek-coder"
        ]

        #api_key = get_api_key("DEEPSEEK_API_KEY", engine)
        if not api_key or api_key == "1234":
            logger.warning("Invalid DeepSeek API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": CONTENT_TYPE_JSON
        }
        api_url = "https://api.deepseek.com/v1/models"  # Adjust URL if needed
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            logger.info(f"Successfully fetched {len(api_models)} models from DeepSeek API")

            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            logger.error(f"Failed to fetch models from DeepSeek: {e}")
            logger.warning(f"Returning fallback list of {len(fallback_models)} DeepSeek models")
            return fallback_models

    elif engine == "lmstudio":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            logger.debug(f"Attempting to connect to {api_url}")
            response = requests.get(api_url, timeout=MODEL_LIST_TIMEOUT)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data["data"]]
                return models
            else:
                logger.warning(f"Failed to fetch models from LM Studio. Status code: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to LM Studio server: {e}")
            return []

    elif engine == "textgen":
        api_url = f"http://{base_ip}:{port}/v1/internal/model/list"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = response.json()["model_names"]
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from text-generation-webui: {e}")
            return []

    elif engine == "kobold":
        api_url = f"http://{base_ip}:{port}/api/v1/model"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            model = response.json()["result"]
            return [model]
        except Exception as e:
            logger.error(f"Failed to fetch models from Kobold: {e}")
            return []

    elif engine == "llamacpp":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = [model["id"] for model in response.json()["data"]]
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from llama.cpp: {e}")
            return []

    elif engine == "vllm":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            # Adapt this based on vLLM"s actual API response structure
            models = [model["id"] for model in response.json()["data"]]
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from vLLM: {e}")
            return []

    elif engine == "openai":
        fallback_models = [
            # GPT-4o Models
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-audio-preview",
            "gpt-4o-audio-preview-2024-10-01",
            "gpt-4o-audio-preview-2024-12-17",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini-audio-preview",
            "gpt-4o-mini-audio-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
            "gpt-4o-realtime-preview",
            "gpt-4o-realtime-preview-2024-10-01",
            "gpt-4o-realtime-preview-2024-12-17",

            # GPT-4 Models
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",

            # GPT-3.5 Models
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",

            # DALL-E Models
            "dall-e-2",
            "dall-e-3",

            # Whisper Models
            "whisper-1",
            "whisper-I",

            # TTS Models
            "tts-1",
            "tts-1-1106",
            "tts-1-hd",
            "tts-1-hd-1106",
            "tts-l-hd",

            # Embedding Models
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",

            # Specialized Models
            "babbage-002",
            "chatgpt-4o-latest",
            "davinci-002",
            "gpt40-0806-loco-vm",

            # O1 Models
            "o1",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",

            # Omni Moderation
            "omni-moderation-2024-09-26",
            "omni-moderation-latest",

            # Future/Experimental
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
            "o3-mini",
            "o3-mini-2025-01-31"
        ]

        #api_key = get_api_key("OPENAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            logger.warning("Invalid OpenAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": CONTENT_TYPE_JSON
        }
        api_url = "https://api.openai.com/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            logger.info(f"Successfully fetched {len(api_models)} models from OpenAI API")

            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            logger.error(f"Failed to fetch models from OpenAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                logger.debug(f"Response status code: {e.response.status_code}")
                logger.debug(f"Response content: {e.response.text}")
            logger.warning(f"Returning fallback list of {len(fallback_models)} OpenAI models")
            return fallback_models

    elif engine == "xai":
        fallback_models = [
            "grok-2",
            "grok-2-1212",
            "grok-2-latest",
            "grok-2-vision",
            "grok-2-vision-1212",
            "grok-2-vision-latest",
            "grok-beta",
            "grok-vision-beta"
        ]

        #api_key = get_api_key("XAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            logger.warning("Invalid XAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": CONTENT_TYPE_JSON
        }
        api_url = "https://api.x.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            logger.info(f"Successfully fetched {len(api_models)} models from XAI API")

            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            logger.error(f"Failed to fetch models from XAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                logger.debug(f"Response status code: {e.response.status_code}")
                logger.debug(f"Response content: {e.response.text}")
            logger.warning(f"Returning fallback list of {len(fallback_models)} XAI models")
            return fallback_models

    elif engine == "mistral":
        fallback_models = [
            "codestral-2405",
            "codestral-2411-rc5",
            "codestral-2412",
            "codestral-2501",
            "codestral-latest",
            "codestral-mamba-2407",
            "codestral-mamba-latest",
            "ministral-3b-2410",
            "ministral-3b-latest",
            "ministral-8b-2410",
            "ministral-8b-latest",
            "mistral-embed",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-large-2411",
            "mistral-large-latest",
            "mistral-large-pixtral-2411",
            "mistral-medium",
            "mistral-medium-2312",
            "mistral-medium-latest",
            "mistral-moderation-2411",
            "mistral-moderation-latest",
            "mistral-ocr-2503",
            "mistral-ocr-latest",
            "mistral-saba-2502",
            "mistral-saba-latest",
            "mistral-small",
            "mistral-small-2312",
            "mistral-small-2402",
            "mistral-small-2409",
            "mistral-small-2501",
            "mistral-small-latest",
            "mistral-tiny",
            "mistral-tiny-2312",
            "mistral-tiny-2407",
            "mistral-tiny-latest",
            "open-codestral-mamba",
            "open-mistral-7b",
            "open-mistral-nemo",
            "open-mistral-nemo-2407",
            "open-mixtral-8x22b",
            "open-mixtral-8x22b-2404",
            "open-mixtral-8x7b",
            "pixtral-12b",
            "pixtral-12b-2409",
            "pixtral-12b-latest",
            "pixtral-large-2411",
            "pixtral-large-latest"
        ]

        #api_key = get_api_key("MISTRAL_API_KEY", engine)
        if not api_key or api_key == "1234":
            logger.warning("Invalid Mistral API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": CONTENT_TYPE_JSON
        }
        api_url = "https://api.mistral.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            logger.info(f"Successfully fetched {len(api_models)} models from Mistral API")

            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            logger.error(f"Failed to fetch models from Mistral: {e}")
            logger.warning(f"Returning fallback list of {len(fallback_models)} Mistral models")
            return fallback_models

    elif engine == "groq":
        fallback_models = [
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "distil-whisper-large-v3-en",
            "gemma2-9b-it",
            "llama-guard-3-8b",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llama-3.3-70b-specdec",
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "llava-v1.5-7b-4096-preview",
            "mixtral-8x7b-32768",
            "mistral-saba-24b",
            "qwen-2.5-32b",
            "qwen-2.5-coder-32b",
            "qwen-qwq-32b",
            "whisper-large-v3",
            "whisper-large-v3-turbo"
        ]

        #api_key = get_api_key("GROQ_API_KEY", engine)
        if not api_key or api_key == "1234":
            logger.warning("Invalid GROQ API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": CONTENT_TYPE_JSON
        }
        api_url = "https://api.groq.com/openai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            logger.info(f"Successfully fetched {len(api_models)} models from GROQ API")

            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            logger.error(f"Failed to fetch models from GROQ: {e}")
            logger.warning(f"Returning fallback list of {len(fallback_models)} GROQ models")
            return fallback_models

    elif engine == "anthropic":
        return [
            "claude-3-5-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-latest",
            "claude-3-5-haiku-20241022"
        ]

    elif engine == "gemini":
        return [
            "learnlrn-1.5-pro-experimental",
            "gemini-2.0-flash-thinking-exp-1219",
            "gemini-2.0-flash-exp",
            "gemini-exp-1206",
            "gemini-exp-1121",
            "gemini-exp-1114",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-latest",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    elif engine == "sentence_transformers":
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "avsolatorio/GIST-small-Embedding-v0",
        ]

    elif engine == "transformers":
        # Standard list of transformers models to show
        fallback_models = [
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",  # Default model we want to use
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "Qwen/QwQ-32B-AWQ",  # Keep QwQ-32B-AWQ model
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-72B-Instruct"
        ]

        # Get list of models from LLM directory
        try:
            # Get models directory dynamically
            try:
                import folder_paths
                models_dir = folder_paths.models_dir
            except (ImportError, AttributeError):
                # Fallback to a default location if folder_paths is not available
                models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                os.makedirs(models_dir, exist_ok=True)
                logger.warning(f"Could not import folder_paths.models_dir, using fallback: {models_dir}")

            llm_path = os.path.join(models_dir, "LLM")
            if os.path.exists(llm_path) and os.path.isdir(llm_path):
                # List directories in the LLM folder
                local_models = []
                for model_dir in os.listdir(llm_path):
                    model_path = os.path.join(llm_path, model_dir)
                    if os.path.isdir(model_path):
                        # Check if it has config.json to verify it's a model
                        if os.path.exists(os.path.join(model_path, "config.json")):
                            if "/" not in model_dir and "\\" not in model_dir:
                                # For non-namespaced models, use the directory name
                                local_models.append(model_dir)
                            else:
                                # For models with namespaces, keep the structure
                                local_models.append(model_dir)

                # If we found local models, add them to our list
                if local_models:
                    logger.info(f"Found {len(local_models)} local transformers models")
                    # Combine local models with fallback models (local models first)
                    combined_models = list(dict.fromkeys(local_models + fallback_models))
                    return combined_models
        except Exception as e:
            logger.error(f"Error scanning local models directory: {e}")

        # If we couldn't find local models, return fallback list
        return fallback_models

    else:
        logger.error(f"Unsupported engine - {engine}")
        return []


def validate_models(model, provider, model_type, base_ip, port, api_key):
    available_models = get_models(provider, base_ip, port, api_key)
    if available_models is None or model not in available_models:
        error_message = f"Invalid {model_type} model selected: {model} for provider {provider}. Available models: {available_models}"
        logger.error(error_message)
        raise ValueError(error_message)


def get_huggingface_url(model_or_url):
    """Convert model name to full HuggingFace API URL if needed"""
    if model_or_url.startswith(('http://', 'https://')):
        return model_or_url
    return f'https://api-inference.huggingface.co/models/{model_or_url}'


def send_huggingface_request(endpoint, payload, api_key, max_retries=3):
    """Send request to HuggingFace Inference API with retry logic"""
    import time
    headers = {"Authorization": f"Bearer {api_key}"}
    url = get_huggingface_url(endpoint)

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                return response

            elif 'estimated_time' in response.text:
                # Handle model loading
                estimated_time = response.json().get('estimated_time', 30)
                logger.info(f"Model loading, waiting {estimated_time} seconds...")
                time.sleep(estimated_time)
                continue

            else:
                raise Exception(f"HuggingFace API error: {response.text}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
