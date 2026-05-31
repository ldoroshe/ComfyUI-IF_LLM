"""Global aiohttp connection pool manager.

Provides a singleton ClientSession shared across all providers
to maximize connection reuse and reduce overhead.
"""

from __future__ import annotations

import aiohttp
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from if_llm.constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

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
