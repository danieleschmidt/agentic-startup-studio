"""Prometheus-based metrics helpers for tracking LLM usage costs."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator

from prometheus_client import Counter, Summary, start_http_server

start_http_server(9102)  # expose metrics on localhost:9102

TOKENS_SPENT = Counter("llm_tokens_total", "Total LLM tokens spent")
_OPERATION_SECONDS = Summary(
    "operation_seconds_total", "Time spent in labelled operations", ["name"]
)


@contextlib.contextmanager
def counter(name: str) -> Generator[None, None, None]:
    """Record the time spent in a named operation."""

    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        _OPERATION_SECONDS.labels(name=name).observe(duration)
