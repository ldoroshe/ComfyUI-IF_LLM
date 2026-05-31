"""Shared utilities for the if_llm package.

These replace duplicated patterns across provider files and send_request.py.
"""
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_base64_prefix(data: str) -> str:
    """Ensure a base64 image string has the data:image prefix."""
    if data.startswith("data:"):
        return data
    return f"data:image/jpeg;base64,{data}"


def is_base64_string(s: str) -> bool:
    """Check if a string is valid base64."""
    try:
        return bool(base64.b64encode(base64.b64decode(s)) == s.encode())
    except Exception:
        return False


def sanitize_error(error_msg: str) -> str:
    """Sanitize error messages for API responses.

    Removes internal details that shouldn't be exposed to users.
    """
    if len(error_msg) > 500:
        error_msg = error_msg[:497] + "..."
    return error_msg


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Override values take precedence."""
    result = base.copy()
    result.update(override)
    return result
