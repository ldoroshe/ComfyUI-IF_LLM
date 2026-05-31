import pytest
from if_llm.providers.connection_pool import get_session, close_pool


@pytest.fixture(autouse=True)
async def cleanup_pool():
    """Clean up pool between tests."""
    yield
    await close_pool()


@pytest.mark.asyncio
async def test_get_session_creates_new():
    session = await get_session()
    assert session is not None
    assert not session.closed


@pytest.mark.asyncio
async def test_get_session_returns_same_instance():
    s1 = await get_session()
    s2 = await get_session()
    assert s1 is s2


@pytest.mark.asyncio
async def test_close_pool_resets():
    s1 = await get_session()
    await close_pool()
    s2 = await get_session()
    assert s1 is not s2
