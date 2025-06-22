"""OpenTelemetry tracing setup for the API server."""
from __future__ import annotations

import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(app: FastAPI, service_name: str = "startup-studio-api") -> None:
    """Initialize OpenTelemetry tracing for a FastAPI app."""
    if os.getenv("ENABLE_TRACING", "false").lower() != "true":
        return

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name})
    )
    exporter = OTLPSpanExporter()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)

