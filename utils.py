"""Backward-compatible re-exports for utils functions.

This module re-exports all utility functions from focused modules
for backward compatibility with existing imports.

New code should import directly from:
    - if_llm.image_utils (image processing/conversion)
    - if_llm.text_utils (text cleaning)
    - if_llm.model_utils (API keys, model listing)
    - if_llm.settings_utils (YAML, combo settings)
    - if_llm.gemini2_utils (Gemini 2.0 specific helpers)
    - if_llm.utils (base64, error handling - already existed)
"""
# Image utilities
from if_llm.image_utils import (
    resize_image_max_side,
    prepare_batch_images,
    process_auto_mode_images,
    convert_images_for_api,
    convert_single_image,
    load_placeholder_image,
    process_images_for_comfy,
    process_mask,
    convert_mask_to_grayscale_alpha,
    tensor_to_base64,
    tensor_to_pil,
    pil_to_tensor,
    base64_to_pil,
    pil_image_to_base64,
)

# Text utilities
from if_llm.text_utils import (
    clean_text,
)

# Model/API utilities
from if_llm.model_utils import (
    get_api_key,
    validate_huggingface_token,
    get_models,
    validate_models,
    get_huggingface_url,
    send_huggingface_request,
)

# Settings utilities
from if_llm.settings_utils import (
    EnhancedYAMLDumper,
    str_presenter,
    numpy_int64_presenter,
    dump_yaml,
    save_combo_settings,
    load_combo_settings,
    create_settings_from_ui,
    format_response,
    print_available_models,
)

# Gemini 2.0 utilities
from if_llm.gemini2_utils import (
    gemini2_process_images,
    gemini2_prepare_response,
    gemini2_create_client,
    validate_gemini_key,
)

# Base utils (already in if_llm.utils)
from if_llm.utils import (
    ensure_base64_prefix,
    is_base64_string,
    sanitize_error,
    merge_dicts,
)

__all__ = [
    # Image utilities
    "resize_image_max_side",
    "prepare_batch_images",
    "process_auto_mode_images",
    "convert_images_for_api",
    "convert_single_image",
    "load_placeholder_image",
    "process_images_for_comfy",
    "process_mask",
    "convert_mask_to_grayscale_alpha",
    "tensor_to_base64",
    "tensor_to_pil",
    "pil_to_tensor",
    "base64_to_pil",
    "pil_image_to_base64",
    # Text utilities
    "clean_text",
    # Model/API utilities
    "get_api_key",
    "validate_huggingface_token",
    "get_models",
    "validate_models",
    "get_huggingface_url",
    "send_huggingface_request",
    # Settings utilities
    "EnhancedYAMLDumper",
    "str_presenter",
    "numpy_int64_presenter",
    "dump_yaml",
    "save_combo_settings",
    "load_combo_settings",
    "create_settings_from_ui",
    "format_response",
    "print_available_models",
    # Gemini 2.0 utilities
    "gemini2_process_images",
    "gemini2_prepare_response",
    "gemini2_create_client",
    "validate_gemini_key",
    # Base utils
    "ensure_base64_prefix",
    "is_base64_string",
    "sanitize_error",
    "merge_dicts",
]
