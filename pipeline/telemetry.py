"""OpenTelemetry tracing and Prometheus metrics setup for the API server."""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
PIPELINE_QUEUE_SIZE = Gauge('pipeline_queue_size', 'Number of items in pipeline queue')
PROCESSING_TIME = Histogram('pipeline_processing_time_seconds', 'Pipeline processing time', ['stage'])
ERROR_COUNT = Counter('pipeline_errors_total', 'Total pipeline errors', ['stage', 'error_type'])


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


def setup_metrics(app: FastAPI) -> None:
    """Set up Prometheus metrics collection for FastAPI app."""

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()

        # Increment active connections
        ACTIVE_CONNECTIONS.inc()

        try:
            response = await call_next(request)

            # Record request metrics
            method = request.method
            endpoint = request.url.path
            status = response.status_code

            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)

            return response

        except Exception as e:
            ERROR_COUNT.labels(stage="http", error_type=type(e).__name__).inc()
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()

    @app.get("/metrics")
    async def metrics_endpoint():
        """Expose Prometheus metrics."""
        return generate_latest()


def record_pipeline_metrics(stage: str, processing_time: float, queue_size: int | None = None) -> None:
    """Record pipeline-specific metrics."""
    PROCESSING_TIME.labels(stage=stage).observe(processing_time)
    if queue_size is not None:
        PIPELINE_QUEUE_SIZE.set(queue_size)


def record_pipeline_error(stage: str, error_type: str) -> None:
    """Record pipeline error metrics."""
    ERROR_COUNT.labels(stage=stage, error_type=error_type).inc()


def init_observability(app: FastAPI, service_name: str = "startup-studio-api") -> None:
    """Initialize complete observability stack (tracing + metrics)."""
    init_tracing(app, service_name)
    setup_metrics(app)

