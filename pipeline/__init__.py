"""
Agentic Startup Studio Data Pipeline

This package implements a comprehensive data pipeline for startup idea validation
through automated founder→investor→smoke-test workflows with quality gates.

Phase 1: Data Ingestion
- Idea capture and validation
- Duplicate detection using pgvector similarity
- Secure storage with audit trails
- CLI interface for management

Author: Agentic Startup Studio
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Agentic Startup Studio"

# Package metadata
__all__ = [
    "models",
    "config",
    "ingestion",
    "storage",
    "cli"
]
