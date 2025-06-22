import asyncio
from collections.abc import Callable, Iterable
from typing import Any


async def async_retry(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Iterable[type[Exception]] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Execute a coroutine with exponential backoff retries."""
    delay = base_delay
    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except tuple(exceptions):
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2

