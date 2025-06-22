import asyncio

import pytest

from pipeline.utils.backoff import async_retry


@pytest.mark.asyncio
async def test_async_retry_success(monkeypatch):
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("fail")
        return "ok"

    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await async_retry(flaky, retries=3, base_delay=0.1, exceptions=(ValueError,))

    assert result == "ok"
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_async_retry_exhaust(monkeypatch):
    async def always_fail():
        raise RuntimeError("boom")

    async def no_sleep(*_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError):
        await async_retry(always_fail, retries=2, base_delay=0.1, exceptions=(RuntimeError,))

