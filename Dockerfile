# =============================================================================
# Multi-stage production Dockerfile for Agentic Startup Studio
# Optimized for security, performance, and minimal attack surface
# =============================================================================

# Build stage - Python dependencies and compilation
FROM python:3.11-slim-bullseye AS builder

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

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
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
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies in virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools && \
    /opt/venv/bin/pip install -r requirements.txt

# Copy source code
COPY --chown=appuser:appuser . .

# Install the application
RUN /opt/venv/bin/pip install -e .

# =============================================================================
# Production stage - Minimal runtime image
# =============================================================================

FROM python:3.11-slim-bullseye AS production

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

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    TIMEOUT=30 \
    KEEPALIVE=2 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Create application directories
RUN mkdir -p /app /app/logs /app/uploads /app/cache /app/temp && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --from=builder --chown=appuser:appuser /app /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9102

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

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