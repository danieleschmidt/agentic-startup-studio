from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from pipeline.infrastructure import (
    get_infrastructure_health,
    get_infrastructure_metrics,
)

app = FastAPI(title="Agentic Startup Studio API")

# Prometheus metrics
health_gauge = Gauge(
    "startup_studio_health_status",
    "1 for healthy, 0 for degraded, -1 for unhealthy",
)


@app.get("/health")
async def health() -> dict:
    """Return overall system health."""
    status = await get_infrastructure_health()
    gauge_value = 1 if status.get("status") == "healthy" else 0
    if status.get("status") == "unhealthy":
        gauge_value = -1
    health_gauge.set(gauge_value)
    return status


@app.get("/metrics")
async def metrics() -> Response:
    """Return Prometheus metrics."""
    # Update metrics from infrastructure collector
    await get_infrastructure_metrics()
    content = generate_latest()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import os
    import uvicorn

    # Secure host binding - use environment variable or default to localhost
    host = os.getenv("HOST_INTERFACE", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
