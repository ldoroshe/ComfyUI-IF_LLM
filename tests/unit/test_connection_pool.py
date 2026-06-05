import pytest

from if_llm.providers.connection_pool import close_pool, get_session, session


@pytest.fixture(autouse=True)
async def cleanup_pool():
    """Clean up pool between tests."""
    yield
    await close_pool()


@pytest.mark.asyncio
async def test_get_session_creates_new():
    session_obj = await get_session()
    assert session_obj is not None
    assert not session_obj.closed


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


@pytest.mark.asyncio
async def test_closed_session_creates_new():
    s1 = await get_session()
    await s1.close()
    assert s1.closed
    s2 = await get_session()
    assert not s2.closed


@pytest.mark.asyncio
async def test_session_context_manager():
    async with session() as s:
        assert s is not None
        assert not s.closed


@pytest.mark.asyncio
async def test_session_context_manager_reuses():
    async with session() as s1:
        async with session() as s2:
            assert s1 is s2
