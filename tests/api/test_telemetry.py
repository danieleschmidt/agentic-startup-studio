from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from pipeline.telemetry import init_tracing


def test_init_tracing_enables_instrumentation(monkeypatch):
    app = FastAPI()
    monkeypatch.setenv("ENABLE_TRACING", "true")
    init_tracing(app)
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)

