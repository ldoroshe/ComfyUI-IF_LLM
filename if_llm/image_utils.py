"""Image processing and conversion utilities for ComfyUI integration.

Contains functions for tensor/PIL/base64 conversions, mask processing,
image batching, and frame handling.
"""
import base64
import io

import os
import logging
from typing import List

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from torchvision.transforms import functional as TF

import node_helpers
from io import BytesIO

logger = logging.getLogger(__name__)


def resize_image_max_side(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image so its longest side is max_size while maintaining aspect ratio."""
    ratio = max_size / max(img.size)
    if ratio < 1:  # Only resize if image is larger than max_size
        new_size = tuple(int(dim * ratio) for dim in img.size)
        return img.resize(new_size, Image.LANCZOS)
    return img


def prepare_batch_images(images):
    """
    Convert images to list of batches.
    Handles tensor, list, and single image inputs while preserving dimensions.

    Args:
        images: torch.Tensor or list of tensors

    Returns:
        List of image tensors
    """
    try:
        if images is None:
            return []

        if isinstance(images, torch.Tensor):
            # Handle 4D tensor [B,H,W,C] - split into list of [H,W,C]
            if images.dim() == 4:
                return [images[i] for i in range(images.shape[0])]
            # Handle 3D tensor [H,W,C] - wrap in list
            elif images.dim() == 3:
                return [images]
            else:
                raise ValueError(f"Invalid tensor dimensions: {images.dim()}")

        # Handle list input - validate each element
        if isinstance(images, list):
            for i, img in enumerate(images):
                if not isinstance(img, torch.Tensor):
                    raise ValueError(f"Image {i} is not a tensor")
            return images

        # Handle single image
        return [images]

    except Exception as e:
        logger.error(f"Error in prepare_batch_images: {str(e)}")
        return []


def process_auto_mode_images(images, mask=None, batch_size=4):
    """
    Process images and masks for auto mode with proper mask dimensionality handling.

    Args:
        images: Input images tensor [B,H,W,C] or list of tensors
        mask: Mask tensor [B,H,W] or [B,1,H,W] or list of tensors
        batch_size: Maximum size of each batch (default 4)

    Returns:
        Tuple of (image_batches, mask_batches) where each is a list of tensors
    """
    try:
        # Convert images to list format
        if images is None or (isinstance(images, (list, tuple)) and len(images) == 0):
            # Return a tuple of empty lists
            return ([], [])

        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # [B,H,W,C]
                images = [images[i] for i in range(images.shape[0])]
            elif images.dim() == 3:  # [H,W,C]
                images = [images]
            else:
                raise ValueError(f"Invalid image tensor dimensions: {images.dim()}")

        # Split images into batches
        image_batches = []
        current_batch = []

        for img in images:
            if len(current_batch) == batch_size:
                image_batches.append(torch.stack(current_batch))
                current_batch = []
            current_batch.append(img)

        if current_batch:  # Don't forget the last batch
            image_batches.append(torch.stack(current_batch))

        # Process masks
        mask_batches = []

        if mask is not None:
            # Standardize mask format
            if isinstance(mask, torch.Tensor):
                # Handle different mask dimensions
                if mask.dim() == 2:  # [H,W]
                    mask = mask.unsqueeze(0)  # -> [1,H,W]
                elif mask.dim() == 3:  # [B,H,W] or [1,H,W]
                    if mask.shape[0] != len(images):
                        # Broadcast mask to match batch size
                        mask = mask.repeat(len(images), 1, 1)
                elif mask.dim() == 4:  # [B,1,H,W] or similar
                    mask = mask.squeeze(1)  # Remove channel dim -> [B,H,W]

                # Split mask into batches matching image batches
                start_idx = 0
                for img_batch in image_batches:
                    batch_size_val = img_batch.size(0)
                    mask_batch = mask[start_idx:start_idx + batch_size_val]

                    mask_batches.append(mask_batch)
                    start_idx += batch_size_val
            else:
                # Handle list of masks
                mask_list = mask if isinstance(mask, list) else [mask] * len(images)
                start_idx = 0
                for img_batch in image_batches:
                    batch_size_val = img_batch.size(0)
                    mask_slice = mask_list[start_idx:start_idx + batch_size_val]

                    # Convert and stack masks
                    mask_tensors = []
                    for m in mask_slice:
                        if isinstance(m, torch.Tensor):
                            if m.dim() == 2:
                                m = m.unsqueeze(0)  # Add batch dim
                            m = m.unsqueeze(-1)  # Add channel dim at end
                        else:
                            # Convert non-tensor masks
                            m = torch.tensor(m, dtype=torch.float32)
                            if m.dim() == 2:
                                m = m.unsqueeze(0).unsqueeze(-1)
                            elif m.dim() == 3:
                                m = m.unsqueeze(-1)
                        mask_tensors.append(m)

                    mask_batch = torch.stack(mask_tensors)
                    mask_batches.append(mask_batch)
                    start_idx += batch_size_val
        else:
            # Create default masks matching image batches
            for img_batch in image_batches:
                mask_batch = torch.ones((img_batch.size(0), img_batch.size(1),
                                       img_batch.size(2)),
                                       dtype=torch.float32,
                                       device=img_batch.device)
                mask_batches.append(mask_batch)

        return image_batches, mask_batches

    except Exception as e:
        logger.error(f"Error in process_auto_mode_images: {str(e)}")
        raise


def convert_images_for_api(images, target_format='tensor'):
    """
    Convert images to the specified format for API consumption.
    Supports conversion to: tensor, base64, pil
    """
    if images is None:
        return None

    # Handle single tensor input with ComfyUI compatibility
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:  # Single image
            images = images.unsqueeze(0)
        # Permute tensor to ComfyUI format (B, H, W, C) -> (B, C, H, W)
        images = images.permute(0, 3, 1, 2)

        if target_format == 'tensor':
            return images
        elif target_format == 'base64':
            return [tensor_to_base64(img) for img in images]
        elif target_format == 'pil':
            return [TF.to_pil_image(img) for img in images]
        else:
            raise ValueError(f"Unsupported target format for tensor: {target_format}")

    # Handle list of tensors input
    elif isinstance(images, list) and all(isinstance(x, torch.Tensor) for x in images):
        # Filter out tensors with unsupported channel counts
        supported_images = []
        for idx, img in enumerate(images):
            if img.shape[0] in [1, 3]:
                supported_images.append(img)
            elif img.shape[0] > 3:
                logger.warning(f"Skipping tensor at index {idx} with {img.shape[0]} channels.")
            else:
                logger.warning(f"Skipping tensor at index {idx} with unsupported number of channels: {img.shape[0]}")
        if not supported_images:
            raise ValueError("No supported image tensors found in the input list.")

        if target_format == 'tensor':
            return torch.stack(supported_images).permute(0, 3, 1, 2)  # Ensure correct format
        elif target_format == 'base64':
            return [tensor_to_base64(img) for img in supported_images]
        elif target_format == 'pil':
            return [TF.to_pil_image(img) for img in supported_images]
        else:
            raise ValueError(f"Unsupported target format for list of tensors: {target_format}")

    # Handle base64 input
    elif isinstance(images, str) or (isinstance(images, list) and all(isinstance(x, str) for x in images)):
        base64_list = [images] if isinstance(images, str) else images
        if target_format == 'base64':
            return base64_list

        # Convert base64 to PIL first
        pil_images = [base64_to_pil(b64) for b64 in base64_list]
        if target_format == 'pil':
            return pil_images
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in pil_images]
            return torch.stack(tensors).permute(0, 2, 3, 1)  # Convert to ComfyUI format (B,H,W,C)
        else:
            raise ValueError(f"Unsupported target format for base64 input: {target_format}")

    # Handle list of PIL images input
    elif isinstance(images, (list, tuple)) and all(isinstance(x, Image.Image) for x in images):
        if target_format == 'pil':
            return images
        elif target_format == 'base64':
            return [pil_image_to_base64(img) for img in images]
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in images]
            return torch.stack(tensors).permute(0, 2, 3, 1)  # Maintain ComfyUI format
        else:
            raise ValueError(f"Unsupported target format for PIL input: {target_format}")

    # If none of the above conditions are met, attempt to convert using the default method
    else:
        try:
            encoded_images = []
            for img in images:
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Expected PIL.Image, got {type(img)}")
                buffered = BytesIO()
                img.save(buffered, format="PNG")  # Adjust format if needed
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                encoded_images.append(img_str)
            return encoded_images
        except Exception as e:
            raise ValueError(f"Unsupported image format or target format: {target_format}. Error: {str(e)}") from e


def convert_single_image(image, target_format):
    """Helper function to convert a single image"""
    if isinstance(image, str) and image.startswith('data:image'):
        # Convert base64 to PIL
        base64_data = image.split('base64,')[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))

    if target_format == 'pil':
        return image
    elif target_format == 'tensor':
        return pil_to_tensor(image)
    elif target_format == 'base64':
        return pil_image_to_base64(image)


def load_placeholder_image(placeholder_image_path):
    # Ensure the placeholder image exists
    if not os.path.exists(placeholder_image_path):
        # Create a proper RGB placeholder image
        placeholder = Image.new('RGB', (512, 512), color=(73, 109, 137))
        os.makedirs(os.path.dirname(placeholder_image_path), exist_ok=True)
        placeholder.save(placeholder_image_path)

    img = node_helpers.pillow(Image.open, placeholder_image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def process_images_for_comfy(images, placeholder_image_path=None, response_key='data', field_name='b64_json', field2_name=""):
    """Process images for ComfyUI, ensuring consistent sizes."""

    def _process_single_image(image):
        try:
            if image is None:
                return load_placeholder_image(placeholder_image_path)

            # Handle JSON/API response
            if isinstance(image, dict):
                try:
                    # Only attempt to extract from response if response_key is provided
                    if response_key and response_key in image:
                        items = image[response_key]
                        if isinstance(items, list):
                            for item in items:
                                # Only attempt to get field_name if it's provided
                                if field2_name and field_name:
                                    image_data = item.get(field2_name, {}).get(field_name)
                                elif field_name:
                                    image_data = item.get(field_name)
                                else:
                                    continue

                                if image_data:
                                    # Convert the first valid image found
                                    if isinstance(image_data, str):
                                        if image_data.startswith(('data:image', 'http:', 'https:')):
                                            image = image_data  # Will be handled by URL processing below
                                        else:
                                            # Handle base64 directly
                                            image_data = base64.b64decode(image_data)
                                            image = Image.open(BytesIO(image_data))
                                            break

                    if isinstance(image, dict):
                        logger.warning(f"No valid image found in response under key '{response_key}'")
                        return load_placeholder_image(placeholder_image_path)
                except Exception as e:
                    logger.error(f"Error processing API response: {str(e)}")
                    return load_placeholder_image(placeholder_image_path)

            # Convert various input types to PIL Image
            if isinstance(image, torch.Tensor):
                # Ensure tensor is in correct format [B,H,W,C] or [H,W,C]
                if image.dim() == 4:
                    if image.shape[-1] != 3:  # Wrong channel dimension
                        image = image.squeeze(1)  # Remove channel dim if [B,1,H,W]
                        if image.shape[-1] != 3:  # Still wrong shape
                            image = image.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
                    image = image.squeeze(0)  # Remove batch dim
                elif image.dim() == 3 and image.shape[0] == 3:
                    image = image.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]

                # Convert to numpy and scale to 0-255 range
                image = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image)

            elif isinstance(image, np.ndarray):
                # Handle numpy arrays
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                if image.shape[-1] != 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)

            elif isinstance(image, str):
                if image.startswith('data:image'):
                    base64_data = image.split('base64,')[1]
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                elif image.startswith(('http:', 'https:')):
                    import requests
                    response = requests.get(image)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(image).convert('RGB')

            # Ensure we have a PIL Image at this point
            if not isinstance(image, Image.Image):
                raise ValueError(f"Failed to convert to PIL Image: {type(image)}")

            # Convert PIL to tensor in ComfyUI format
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)

            # Ensure NHWC format
            if img_tensor.dim() == 3:  # [H,W,C]
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: [1,H,W,C]

            # Create mask
            mask_tensor = torch.ones((1, img_tensor.shape[1], img_tensor.shape[2]),
                                     dtype=torch.float32)

            return img_tensor, mask_tensor

        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            return load_placeholder_image(placeholder_image_path)

    try:
        # Handle API responses
        if isinstance(images, dict) and response_key in images:
            # Process each item in API response
            all_tensors = []
            all_masks = []

            items = images[response_key]
            if isinstance(items, list):
                for item in items:
                    try:
                        img_tensor, mask_tensor = _process_single_image({response_key: [item]})
                        all_tensors.append(img_tensor)
                        all_masks.append(mask_tensor)
                    except Exception as e:
                        logger.error(f"Error processing response item: {str(e)}")
                        continue

                if all_tensors:
                    return torch.cat(all_tensors, dim=0), torch.cat(all_masks, dim=0)

            # If no valid images processed, return placeholder
            return load_placeholder_image(placeholder_image_path)

        # Handle list/batch of images
        if isinstance(images, (list, tuple)):
            all_tensors = []
            all_masks = []

            for img in images:
                try:
                    img_tensor, mask_tensor = _process_single_image(img)
                    all_tensors.append(img_tensor)
                    all_masks.append(mask_tensor)
                except Exception as e:
                    logger.error(f"Error processing batch image: {str(e)}")
                    continue

            if all_tensors:
                return torch.cat(all_tensors, dim=0), torch.cat(all_masks, dim=0)

            return load_placeholder_image(placeholder_image_path)

        # Handle single image
        return _process_single_image(images)

    except Exception as e:
        logger.error(f"Error in process_images_for_comfy: {str(e)}")
        return _process_single_image(None)


def process_mask(retrieved_mask, image_tensor):
    """
    Process the retrieved_mask to ensure it's in the correct format.
    The mask should be a tensor of shape (B, H, W), matching image_tensor's batch size and dimensions.
    """
    try:
        # Handle torch.Tensor
        if isinstance(retrieved_mask, torch.Tensor):
            # Normalize dimensions
            if retrieved_mask.dim() == 2:  # (H, W)
                retrieved_mask = retrieved_mask.unsqueeze(0)  # Add batch dimension
            elif retrieved_mask.dim() == 3:
                if retrieved_mask.shape[0] != image_tensor.shape[0]:
                    # Adjust batch size
                    retrieved_mask = retrieved_mask.repeat(image_tensor.shape[0], 1, 1)
            elif retrieved_mask.dim() == 4:
                # If mask has a channel dimension, reduce it
                retrieved_mask = retrieved_mask.squeeze(1)
            else:
                raise ValueError(f"Invalid mask tensor dimensions: {retrieved_mask.shape}")

            # Ensure proper format
            retrieved_mask = retrieved_mask.float()
            if retrieved_mask.max() > 1.0:
                retrieved_mask = retrieved_mask / 255.0

            # Ensure mask dimensions match image dimensions
            if retrieved_mask.shape[1:] != image_tensor.shape[2:]:
                # Resize mask to match image dimensions
                retrieved_mask = torch.nn.functional.interpolate(
                    retrieved_mask.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return retrieved_mask

        # Handle PIL Image
        elif isinstance(retrieved_mask, Image.Image):
            mask_array = np.array(retrieved_mask.convert('L')).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle numpy array
        elif isinstance(retrieved_mask, np.ndarray):
            mask_array = retrieved_mask.astype(np.float32)
            if mask_array.max() > 1.0:
                mask_array = mask_array / 255.0
            if mask_array.ndim == 2:
                pass  # (H, W)
            elif mask_array.ndim == 3:
                mask_array = np.mean(mask_array, axis=2)  # Convert to grayscale
            else:
                raise ValueError(f"Invalid mask array dimensions: {mask_array.shape}")

            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle other types (e.g., file paths, base64 strings)
        elif isinstance(retrieved_mask, str):
            # Attempt to process as file path or base64 string
            if os.path.exists(retrieved_mask):
                pil_image = Image.open(retrieved_mask).convert('L')
            elif retrieved_mask.startswith('data:image'):
                base64_data = retrieved_mask.split('base64,')[1]
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_data)).convert('L')
            else:
                raise ValueError(f"Invalid mask string: {retrieved_mask}")
            return process_mask(pil_image, image_tensor)

        else:
            raise ValueError(f"Unsupported mask type: {type(retrieved_mask)}")

    except Exception as e:
        logger.error(f"Error processing mask: {str(e)}")
        # Return a default mask matching the image dimensions
        return torch.ones((image_tensor.shape[0], image_tensor.shape[2], image_tensor.shape[3]), dtype=torch.float32)


def convert_mask_to_grayscale_alpha(mask_input):
    """
    Convert mask to grayscale alpha channel.
    Handles tensors, PIL images and numpy arrays.
    Returns tensor in shape [B,1,H,W].
    """
    if isinstance(mask_input, torch.Tensor):
        # Handle tensor input
        if mask_input.dim() == 2:  # [H,W]
            return mask_input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif mask_input.dim() == 3:  # [C,H,W] or [B,H,W]
            if mask_input.shape[0] in [1, 3, 4]:  # Assume channel-first
                if mask_input.shape[0] == 4:  # Use alpha channel
                    return mask_input[3:4].unsqueeze(0)
                else:  # Convert to grayscale
                    weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                    return (mask_input * weights.view(-1, 1, 1)).sum(0).unsqueeze(0).unsqueeze(0)
        elif mask_input.dim() == 4:  # [B,C,H,W]
            if mask_input.shape[1] == 4:  # Use alpha channel
                return mask_input[:, 3:4]
            else:  # Convert to grayscale
                weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                return (mask_input * weights.view(1, -1, 1, 1)).sum(1).unsqueeze(1)

    elif isinstance(mask_input, Image.Image):
        # Convert PIL image to grayscale
        mask = mask_input.convert('L')
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    elif isinstance(mask_input, np.ndarray):
        # Handle numpy array
        if mask_input.ndim == 2:  # [H,W]
            tensor = torch.from_numpy(mask_input).float()
            return tensor.unsqueeze(0).unsqueeze(0)
        elif mask_input.ndim == 3:  # [H,W,C]
            if mask_input.shape[2] == 4:  # Use alpha channel
                tensor = torch.from_numpy(mask_input[:, :, 3]).float()
            else:  # Convert to grayscale
                tensor = torch.from_numpy(np.dot(mask_input[..., :3], [0.299, 0.587, 0.114])).float()
            return tensor.unsqueeze(0).unsqueeze(0)

    raise ValueError(f"Unsupported mask input type: {type(mask_input)}")


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a tensor to a base64-encoded PNG image string."""
    try:
        # Ensure the tensor is in [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)

        # Handle different tensor dimensions
        if tensor.dim() == 3:
            # [C, H, W]
            if tensor.shape[0] == 1:
                # Grayscale image, convert to RGB by repeating channels
                image = tensor.squeeze(0).unsqueeze(-1).cpu().numpy()  # [H, W, 1]
                image = np.repeat(image, 3, axis=2)  # [H, W, 3]
            elif tensor.shape[0] == 3:
                # RGB image
                image = tensor.permute(1, 2, 0).cpu().numpy()
            else:
                # Handle tensors with more than 3 channels: select the first 3 channels
                logger.warning(f"Unsupported number of channels: {tensor.shape[0]}. Selecting first 3 channels.")
                if tensor.shape[0] >= 3:
                    image = tensor[:3, :, :].permute(1, 2, 0).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported number of channels: {tensor.shape[0]}")
        elif tensor.dim() == 2:
            # [H, W] Grayscale image
            image = tensor.unsqueeze(-1).cpu().numpy()
            image = np.repeat(image, 3, axis=2)
        else:
            raise ValueError(f"Unsupported tensor shape for conversion: {tensor.shape}")

        # Convert to uint8
        image = (image * 255).astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(image)

        # Save image to buffer
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.error(f"Error converting tensor to base64: {str(e)}", exc_info=True)
        raise


def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image with better error handling and format detection.

    Args:
        tensor: A PyTorch tensor representing an image

    Returns:
        PIL.Image: The converted PIL image
    """
    try:
        # Ensure tensor is on CPU
        tensor = tensor.cpu()

        # Handle different tensor shapes
        if tensor.dim() == 4 and tensor.shape[0] == 1:  # [1, C, H, W] or [1, H, W, C]
            tensor = tensor.squeeze(0)  # Remove batch dimension

        # Determine if we have a channels-first or channels-last format
        if tensor.dim() == 3:
            # Handle both [C, H, W] and [H, W, C] formats
            if tensor.shape[0] in [1, 3, 4]:  # Channels-first format [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # Convert to [H, W, C]

        # Special case for grayscale
        if tensor.dim() == 2:
            # Add a channel dimension for grayscale [H, W] -> [H, W, 1]
            tensor = tensor.unsqueeze(-1)

        # Convert to numpy array
        tensor_np = tensor.numpy()

        # Scale to 0-255 range for uint8
        tensor_np = np.clip(tensor_np * 255, 0, 255).astype(np.uint8)

        # Create PIL image
        pil_image = Image.fromarray(tensor_np)
        return pil_image

    except Exception as e:
        logger.error(f"Error in tensor_to_pil: {e}")
        raise ValueError(f"Failed to convert tensor to PIL image: {e}")


def pil_to_tensor(pil_image):
    # Convert PIL image to tensor
    tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
    return tensor.permute(2, 0, 1) if tensor.dim() == 3 else tensor.unsqueeze(0)


def base64_to_pil(base64_str):
    """Convert base64 string to PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split('base64,')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def pil_image_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL Image to a data URL."""
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to data URL: {str(e)}", exc_info=True)
        raise
