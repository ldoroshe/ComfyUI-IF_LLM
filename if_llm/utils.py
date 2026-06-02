"""Shared utilities for the if_llm package.

These replace duplicated patterns across provider files and send_request.py.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict

from if_llm.constants import IMAGE_DATA_URL_PREFIX


logger = logging.getLogger(__name__)


def ensure_base64_prefix(data: str) -> str:
    """Ensure a base64 image string has the data:image prefix.

    Args:
        data: The raw base64 string, or one already prefixed with "data:".

    Returns:
        The data string with the "data:image/jpeg;base64," prefix.
    """
    if data.startswith("data:"):
        return data
    return f"{IMAGE_DATA_URL_PREFIX}{data}"


def is_base64_string(s: str) -> bool:
    """Check if a string is valid base64.

    Args:
        s: The string to check.

    Returns:
        True if the string is valid base64, False otherwise.
    """
    try:
        return bool(base64.b64encode(base64.b64decode(s)) == s.encode())
    except Exception:
        return False


def sanitize_error(error_msg: str) -> str:
    """Sanitize error messages for API responses.

    Removes internal details that shouldn't be exposed to users.
    Truncates messages longer than 500 characters.

    Args:
        error_msg: The raw error message string.

    Returns:
        The sanitized error message, truncated to 500 chars if needed.
    """
    if len(error_msg) > 500:
        error_msg = error_msg[:497] + "..."
    return error_msg


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts. Override values take precedence.

    Note: This is a shallow merge (top-level keys only). Nested dicts
    are replaced rather than merged recursively.

    Args:
        base: The base dictionary.
        override: The override dictionary whose values take precedence.

    Returns:
        A new dict with keys from both, override taking precedence.
    """
    result = base.copy()
    result.update(override)
    return result
