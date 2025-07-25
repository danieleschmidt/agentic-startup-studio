import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import REGISTRY

from pipeline.telemetry import (
    init_tracing, 
    setup_metrics, 
    init_observability,
    record_pipeline_metrics,
    record_pipeline_error,
    REQUEST_COUNT,
    REQUEST_DURATION,
    ACTIVE_CONNECTIONS,
    PIPELINE_QUEUE_SIZE,
    PROCESSING_TIME,
    ERROR_COUNT
)


def test_init_tracing_enables_instrumentation(monkeypatch):
    app = FastAPI()
    monkeypatch.setenv("ENABLE_TRACING", "true")
    init_tracing(app)
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)


def test_init_tracing_disabled_by_default():
    app = FastAPI()
    init_tracing(app)
    # When tracing is disabled, provider should be the default NoOpTracerProvider
    provider = trace.get_tracer_provider()
    assert not isinstance(provider, TracerProvider)


def test_setup_metrics_adds_middleware():
    app = FastAPI()
    setup_metrics(app)
    
    # Check that metrics endpoint was added
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_metrics_middleware_records_requests():
    app = FastAPI()
    setup_metrics(app)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    client = TestClient(app)
    
    # Clear metrics before test
    REQUEST_COUNT._value.clear()
    REQUEST_DURATION._sum.clear()
    
    response = client.get("/test")
    assert response.status_code == 200
    
    # Verify metrics were recorded
    metrics = client.get("/metrics").text
    assert "http_requests_total" in metrics
    assert "http_request_duration_seconds" in metrics


def test_record_pipeline_metrics():
    # Clear metrics before test
    PROCESSING_TIME._sum.clear()
    PIPELINE_QUEUE_SIZE._value.clear()
    
    record_pipeline_metrics("ingestion", 1.5, 42)
    
    # Check that metrics were recorded
    assert PROCESSING_TIME.labels(stage="ingestion")._sum._value > 0
    assert PIPELINE_QUEUE_SIZE._value._value == 42


def test_record_pipeline_error():
    # Clear metrics before test
    ERROR_COUNT._value.clear()
    
    record_pipeline_error("processing", "ValueError")
    
    # Check that error was recorded
    assert ERROR_COUNT.labels(stage="processing", error_type="ValueError")._value._value == 1


def test_init_observability_sets_up_both_tracing_and_metrics(monkeypatch):
    app = FastAPI()
    monkeypatch.setenv("ENABLE_TRACING", "true")
    
    init_observability(app, "test-service")
    
    # Verify tracing is set up
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    
    # Verify metrics endpoint exists
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200


@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry between tests to avoid conflicts."""
    yield
    # Clear all collectors from default registry
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

