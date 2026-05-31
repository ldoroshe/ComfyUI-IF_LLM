"""Gemini 2.0 specific utilities for image processing and API interaction."""
import logging

logger = logging.getLogger(__name__)


def gemini2_process_images(images, max_input_images=5, target_size=(768, 768)):
    """
    Process a batch of images for Gemini 2.0 API.

    Args:
        images (torch.Tensor or list): Image batch in ComfyUI format [B,H,W,C] or list of tensors
        max_input_images (int): Maximum number of images to include (Gemini may have limits)
        target_size (tuple): Target size for images (width, height)

    Returns:
        list: List of processed PIL images ready for the Gemini API
    """
    import torch
    from PIL import Image
    import numpy as np

    # Handle different input types
    processed_images = []

    if isinstance(images, torch.Tensor):
        # Handle 4D tensor [B,H,W,C]
        if images.dim() == 4:
            # Limit to max_input_images
            batch_size = min(images.shape[0], max_input_images)

            for i in range(batch_size):
                # Get single image tensor [H,W,C]
                img_tensor = images[i].cpu()

                # Convert to numpy and scale to 0-255
                img_np = (img_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)

                # Convert to PIL
                pil_img = Image.fromarray(img_np)

                # Resize to target size if needed
                if pil_img.size != target_size:
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

                processed_images.append(pil_img)

        # Handle 3D tensor [H,W,C]
        elif images.dim() == 3:
            img_tensor = images.cpu()
            img_np = (img_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            if pil_img.size != target_size:
                pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

            processed_images.append(pil_img)

    # Handle list of tensors
    elif isinstance(images, list):
        # Limit to max_input_images
        num_images = min(len(images), max_input_images)

        for i in range(num_images):
            img = images[i]

            if isinstance(img, torch.Tensor):
                img_tensor = img.cpu()

                # Handle different tensor dimensions
                if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:  # [1,H,W,C]
                    img_tensor = img_tensor.squeeze(0)

                img_np = (img_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                if pil_img.size != target_size:
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

                processed_images.append(pil_img)

    return processed_images


def gemini2_prepare_response(response, width=512, height=512):
    """
    Extract and prepare images from Gemini 2.0 API response.

    Args:
        response: Gemini API response object
        width (int): Target width for extracted images
        height (int): Target height for extracted images

    Returns:
        tuple: (list of image binaries, response text)
    """
    images = []
    response_text = ""

    # Handle empty response
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return images, "No response generated"

    # Process each candidate
    for candidate in response.candidates:
        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
            continue

        for part in candidate.content.parts:
            # Process text parts
            if hasattr(part, 'text') and part.text:
                response_text += part.text + "\n"

            # Process image parts
            if hasattr(part, 'inline_data') and part.inline_data:
                try:
                    # Get binary image data
                    image_binary = part.inline_data.data
                    images.append(image_binary)
                except Exception as e:
                    logger.error(f"Error extracting image from response: {e}")

    return images, response_text


def gemini2_create_client(api_key):
    """
    Create and return a Gemini API client.

    Args:
        api_key (str): The Gemini API key

    Returns:
        Client: Gemini API client object
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        raise ImportError("The google-generativeai package is required. Please install it with: pip install google-generativeai")
    except Exception as e:
        raise RuntimeError(f"Failed to create Gemini client: {str(e)}")


def validate_gemini_key(api_key):
    """
    Validate a Gemini API key by making a simple test request.

    Args:
        api_key (str): The Gemini API key to validate

    Returns:
        bool: True if key is valid, False otherwise
    """
    try:
        from google import genai

        # Initialize client with the key
        client = genai.Client(api_key=api_key)

        # Try a simple models list request
        models = client.models.list()

        # If we get here, the key is valid
        return True
    except Exception as e:
        logger.error(f"Invalid Gemini API key: {str(e)}")
        return False
