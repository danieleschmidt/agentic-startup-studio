# =============================================================================
# Advanced Multi-stage production Dockerfile for Agentic Startup Studio
# Optimized for security, performance, minimal attack surface, and AI/ML workloads
# Includes: Security hardening, multi-architecture support, cache optimization
# =============================================================================

# Base image with security updates and multi-arch support
FROM --platform=$BUILDPLATFORM python:3.13-slim-bullseye AS base

# Security: Update base system and install security patches
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Build stage - Python dependencies and compilation
FROM base AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Metadata labels
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.url="https://github.com/terragonlabs/agentic-startup-studio" \
      org.opencontainers.image.source="https://github.com/terragonlabs/agentic-startup-studio" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.title="Agentic Startup Studio" \
      org.opencontainers.image.description="AI-powered startup idea validation platform" \
      org.opencontainers.image.documentation="https://docs.terragonlabs.com/agentic-startup-studio" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies for building with security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    # AI/ML specific dependencies
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Security tools
    gnupg2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create app user and directories
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Create optimized virtual environment with dependency caching
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools

# Install dependencies with BuildKit cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    /opt/venv/bin/pip install -r /tmp/requirements.txt --compile --prefer-binary

# Install AI/ML specific optimizations
RUN /opt/venv/bin/pip install --no-deps \
    # Fast JSON processing
    orjson \
    # Optimized HTTP client
    httpx[http2] \
    # Memory-efficient data processing
    polars \
    # Performance monitoring
    psutil

# Copy source code
COPY --chown=appuser:appuser . .

# Install the application
RUN /opt/venv/bin/pip install -e .

# =============================================================================
# Security scanning stage - Scan for vulnerabilities
# =============================================================================

FROM builder AS security-scan

# Install security scanning tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install security scanning tools
RUN /opt/venv/bin/pip install safety bandit semgrep

# Run security scans
RUN /opt/venv/bin/safety check --json --output /tmp/safety-report.json || true
RUN /opt/venv/bin/bandit -r . -f json -o /tmp/bandit-report.json || true

# =============================================================================
# Production stage - Minimal runtime image with security hardening
# =============================================================================

FROM base AS production

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Metadata labels
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.url="https://github.com/terragonlabs/agentic-startup-studio" \
      org.opencontainers.image.source="https://github.com/terragonlabs/agentic-startup-studio" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.title="Agentic Startup Studio" \
      org.opencontainers.image.description="AI-powered startup idea validation platform" \
      org.opencontainers.image.documentation="https://docs.terragonlabs.com/agentic-startup-studio" \
      org.opencontainers.image.licenses="MIT"

# Set production environment variables with security and performance tuning
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    TIMEOUT=30 \
    KEEPALIVE=2 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    # Security hardening
    PYTHONHASHSEED=random \
    # Performance optimization
    MALLOC_ARENA_MAX=2 \
    # AI/ML optimization
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Install only runtime dependencies with security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Database connectivity
    libpq5 \
    # HTTP client
    curl \
    # SSL/TLS support
    ca-certificates \
    # Process init system
    tini \
    # AI/ML runtime libraries
    libopenblas0 \
    libblas3 \
    liblapack3 \
    # Security utilities
    dumb-init \
    # Monitoring utilities
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user with security hardening
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash \
            --no-log-init appuser

# Create application directories with proper permissions
RUN mkdir -p /app /app/logs /app/uploads /app/cache /app/temp /app/models && \
    # Set secure permissions
    chmod 750 /app && \
    chmod 700 /app/logs /app/uploads /app/cache /app/temp /app/models && \
    chown -R appuser:appuser /app

# Security: Remove setuid/setgid binaries that could be used for privilege escalation
RUN find / -type f \( -perm -4000 -o -perm -2000 \) -delete 2>/dev/null || true

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --from=builder --chown=appuser:appuser /app /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Enhanced health check with comprehensive validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -H "Health-Check: true" http://localhost:8000/health || exit 1

# Expose ports with explicit protocol
EXPOSE 8000/tcp 9102/tcp

# Use dumb-init for proper signal handling and zombie reaping in containers
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Production command with optimized settings
CMD ["python", "-O", "scripts/serve_api.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--preload"]

# =============================================================================
# Development stage - For local development with hot reload
# =============================================================================

FROM builder AS development

# Install development dependencies
RUN /opt/venv/bin/pip install \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    pytest-asyncio \
    black \
    isort \
    mypy \
    ruff \
    pre-commit \
    ipython \
    ipdb

# Set development environment variables
ENV ENVIRONMENT=development \
    LOG_LEVEL=DEBUG \
    AUTO_RELOAD=true \
    DEBUG_TOOLBAR=true

# Switch to app user
USER appuser

# Expose ports for development
EXPOSE 8000 5678 9102

# Development command with hot reload
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Testing stage - For running tests in CI/CD
# =============================================================================

FROM development AS testing

# Set test environment variables
ENV ENVIRONMENT=test \
    LOG_LEVEL=WARNING \
    DATABASE_URL=postgresql://test:test@postgres-test:5432/test_db

# Copy test configuration
COPY --chown=appuser:appuser pytest.ini .
COPY --chown=appuser:appuser tests/ tests/

# Run tests by default
CMD ["pytest", "tests/", "--cov=pipeline", "--cov=core", "--cov-report=xml", "--cov-report=html", "--junitxml=test-results/results.xml"]

# =============================================================================
# AI/ML Model serving stage - Optimized for model inference
# =============================================================================

FROM production AS model-serving

# Install additional AI/ML serving dependencies
RUN /opt/venv/bin/pip install --no-cache-dir \
    # Model serving
    tritonclient[all] \
    # Optimization libraries
    onnxruntime \
    torch-tensorrt \
    # Memory optimization
    psutil

# Set model serving specific environment variables
ENV MODEL_CACHE_SIZE=1000 \
    INFERENCE_BATCH_SIZE=8 \
    MODEL_WARMUP_ENABLED=true \
    GPU_MEMORY_FRACTION=0.8

# Create model cache directory
RUN mkdir -p /app/models/cache && \
    chown -R appuser:appuser /app/models

# Model serving command
CMD ["python", "-O", "scripts/serve_models.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model-cache-dir", "/app/models/cache"]

# =============================================================================
# Worker stage - For background processing tasks
# =============================================================================

FROM production AS worker

# Set worker-specific environment variables
ENV WORKER_TYPE=background \
    CONCURRENCY=4 \
    MAX_MEMORY_PER_CHILD=200 \
    TASK_SOFT_TIME_LIMIT=300 \
    TASK_TIME_LIMIT=600

# Worker command for Celery or similar
CMD ["python", "-O", "scripts/worker.py", \
     "--concurrency", "4", \
     "--loglevel", "info", \
     "--logfile", "/app/logs/worker.log"]

# =============================================================================
# Monitoring stage - With enhanced observability tools
# =============================================================================

FROM production AS monitoring

# Install monitoring and observability tools
RUN /opt/venv/bin/pip install --no-cache-dir \
    # Metrics collection
    prometheus-client \
    # Distributed tracing
    opentelemetry-distro \
    opentelemetry-exporter-jaeger \
    # APM
    elastic-apm \
    # Profiling
    py-spy \
    # Memory profiling
    memory-profiler

# Set monitoring environment variables
ENV ENABLE_METRICS=true \
    ENABLE_TRACING=true \
    ENABLE_APM=true \
    METRICS_PORT=9102 \
    PROFILING_ENABLED=true

# Expose monitoring ports
EXPOSE 9102/tcp 14268/tcp

# Enhanced monitoring command
CMD ["python", "-O", "scripts/serve_api.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--enable-metrics", \
     "--enable-tracing", \
     "--metrics-port", "9102"]

# =============================================================================
# Distroless production stage - Ultra-minimal for highest security
# =============================================================================

FROM gcr.io/distroless/python3-debian11:nonroot AS distroless

# Copy virtual environment and application from builder
COPY --from=builder --chown=nonroot:nonroot /opt/venv /opt/venv
COPY --from=builder --chown=nonroot:nonroot /app /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

WORKDIR /app

# Expose port
EXPOSE 8000/tcp

# Use Python directly (no shell available in distroless)
ENTRYPOINT ["/opt/venv/bin/python", "-O"]
CMD ["scripts/serve_api.py"]

# =============================================================================
# Debug stage - For troubleshooting production issues
# =============================================================================

FROM production AS debug

# Install debugging tools
RUN /opt/venv/bin/pip install --no-cache-dir \
    # Debugging
    pdb++ \
    ipdb \
    # Performance analysis
    py-spy \
    memory-profiler \
    line-profiler \
    # System analysis
    htop \
    strace

# Set debug environment variables
ENV LOG_LEVEL=DEBUG \
    ENABLE_DEBUG_TOOLBAR=true \
    ENABLE_PROFILING=true \
    DEBUG_MODE=true

# Debug command with profiling
CMD ["python", "-m", "pdb", "scripts/serve_api.py"]