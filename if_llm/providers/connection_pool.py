"""Global aiohttp connection pool manager.

Provides a singleton ClientSession shared across all providers
to maximize connection reuse and reduce overhead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import aiohttp

from if_llm.constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


async def make_request(
    session: aiohttp.ClientSession,
    api_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> Any:
    """Make an HTTP request using the session.

    Args:
        session: The aiohttp ClientSession.
        api_url: The target API endpoint.
        headers: Request headers.
        payload: Request body as dict.

    Returns:
        The response data (parsed JSON or raw content).
    """
    async with session.post(api_url, headers=headers, json=payload) as resp:
        if resp.status >= 400:
            error_text = await resp.text()
            logger.error(f"Request failed with status {resp.status}: {error_text}")
            raise aiohttp.ClientError(f"HTTP error: {resp.status} - {error_text}")
        return await resp.json()


async def handle_normal_inference(response: Any) -> dict[str, Any]:
    """Handle normal inference response.

    Args:
        response: The raw response from the API.

    Returns:
        A standardized dict with the response data.
    """
    if isinstance(response, dict):
        return response
    elif isinstance(response, list):
        return {"results": response}
    else:
        return {"raw": str(response)}


# Global session pool — one per process
_session: aiohttp.ClientSession | None = None


async def get_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp ClientSession.

    Callers should use get_session() instead of creating new
    aiohttp.ClientSession() instances. This ensures connection
    reuse across all providers.

    Returns:
        The global aiohttp.ClientSession instance, creating one if needed.
    """
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def close_pool() -> None:
    """Close the global session pool. Call on shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


@asynccontextmanager
async def session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Context manager for getting a session.

    Usage:
        async with session() as s:
            async with s.post(url, json=data) as resp:
                ...

    Yields:
        The global aiohttp.ClientSession instance.
    """
    yield await get_session()
