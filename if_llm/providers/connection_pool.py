"""Global aiohttp connection pool manager.

Provides a singleton ClientSession shared across all providers
to maximize connection reuse and reduce overhead.
"""

import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Global session pool — one per process
_session: aiohttp.ClientSession | None = None


async def get_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp ClientSession.

    Callers should use get_session() instead of creating new
    aiohttp.ClientSession() instances. This ensures connection
    reuse across all providers.
    """
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=120)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def close_pool():
    """Close the global session pool. Call on shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


@asynccontextmanager
async def session():
    """Context manager for getting a session.

    Usage:
        async with session() as s:
            async with s.post(url, json=data) as resp:
                ...
    """
    yield await get_session()
