# API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Data Models and Schemas](#data-models-and-schemas)
3. [CLI Interface](#cli-interface)
4. [Service Interfaces](#service-interfaces)
5. [Authentication and Authorization](#authentication-and-authorization)
6. [Rate Limiting and Usage Constraints](#rate-limiting-and-usage-constraints)
7. [Error Codes and Responses](#error-codes-and-responses)
8. [Integration Patterns](#integration-patterns)
9. [Configuration Reference](#configuration-reference)

---

## Overview

The Agentic Startup Studio data pipeline provides programmatic interfaces for managing startup ideas through a complete validation and development workflow. The system exposes functionality through CLI commands, Python service interfaces, and workflow orchestration APIs.

### Core Capabilities

- **Idea Management**: Create, validate, update, and track startup ideas
- **Workflow Orchestration**: Execute multi-stage pipeline processing
- **Budget Tracking**: Monitor costs across all operations
- **Vector Search**: Find similar ideas using semantic similarity
- **Quality Gates**: Validate outputs at each pipeline stage

### API Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ CLI Interface│    │ Service Layer │    │ Data Models     │
│             │◄──►│              │◄──►│                 │
│ Rich Console│    │ Orchestrator │    │ Pydantic Schemas│
└─────────────┘    └──────────────┘    └─────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Input       │    │ LangGraph    │    │ PostgreSQL      │
│ Validation  │    │ Workflows    │    │ + pgvector      │
└─────────────┘    └──────────────┘    └─────────────────┘
```

---

## Data Models and Schemas

### Core Entities

#### Idea Model

The central entity representing a startup idea with complete metadata and tracking.

```python
from pipeline.models.idea import Idea, IdeaStatus, PipelineStage

idea = Idea(
    title="AI-Powered Code Review Assistant",
    description="Automated code review using machine learning...",
    category=IdeaCategory.AI_ML,
    status=IdeaStatus.DRAFT,
    current_stage=PipelineStage.IDEATE,
    problem_statement="Developers spend too much time on manual code reviews",
    solution_description="AI assistant that provides instant feedback...",
    target_market="Software development teams",
    evidence_links=["https://research.example.com/code-review-study"]
)
```

**Field Reference:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `idea_id` | UUID | Auto | Unique identifier (auto-generated) |
| `title` | str | Yes | Idea title (10-200 chars) |
| `description` | str | Yes | Detailed description (10-5000 chars) |
| `category` | IdeaCategory | No | Business category (default: uncategorized) |
| `status` | IdeaStatus | No | Processing status (default: DRAFT) |
| `current_stage` | PipelineStage | No | Pipeline stage (default: IDEATE) |
| `stage_progress` | float | No | Stage completion (0.0-1.0) |
| `problem_statement` | str | No | Problem description (max 1000 chars) |
| `solution_description` | str | No | Solution overview (max 1000 chars) |
| `target_market` | str | No | Target customer segment (max 500 chars) |
| `evidence_links` | List[str] | No | Supporting evidence URLs |
| `created_at` | datetime | Auto | Creation timestamp (UTC) |
| `updated_at` | datetime | Auto | Last modification timestamp (UTC) |
| `created_by` | str | No | User identifier (max 100 chars) |
| `deck_path` | str | No | Generated pitch deck path |
| `research_data` | Dict | No | Research artifacts and scores |
| `investor_scores` | Dict | No | Investor evaluation results |

#### Enumeration Types

**IdeaStatus**: Current processing state
- `DRAFT` - Initial creation state
- `VALIDATING` - Under validation review
- `VALIDATED` - Passed validation checks
- `REJECTED` - Failed validation or quality gates
- `RESEARCHING` - Evidence collection phase
- `BUILDING` - MVP development phase
- `TESTING` - Smoke test execution
- `DEPLOYED` - Live MVP deployed
- `ARCHIVED` - Permanently stored/inactive

**PipelineStage**: Workflow progression tracking
- `IDEATE` - Initial idea capture
- `RESEARCH` - Evidence collection and analysis
- `DECK` - Pitch deck generation
- `INVESTORS` - Investor evaluation
- `MVP` - MVP development
- `SMOKE_TEST` - Market validation testing
- `COMPLETE` - Pipeline finished

**IdeaCategory**: Business domain classification
- `FINTECH` - Financial technology
- `HEALTHTECH` - Healthcare technology
- `EDTECH` - Educational technology
- `SAAS` - Software as a Service
- `ECOMMERCE` - Electronic commerce
- `AI_ML` - Artificial Intelligence/Machine Learning
- `BLOCKCHAIN` - Blockchain/Cryptocurrency
- `CONSUMER` - Consumer applications
- `ENTERPRISE` - Enterprise software
- `MARKETPLACE` - Multi-sided platforms
- `UNCATEGORIZED` - Unclassified ideas

### Input/Output Models

#### IdeaDraft

Input model for creating new ideas with validation.

```python
from pipeline.models.idea import IdeaDraft

draft = IdeaDraft(
    title="Revolutionary Food Delivery App",
    description="AI-optimized delivery routing with sustainability focus...",
    category="consumer",
    problem_statement="Current delivery is inefficient and environmentally harmful",
    solution_description="Machine learning optimizes routes for speed and emissions",
    target_market="Environmentally conscious urban consumers",
    evidence_links=["https://study.example.com/delivery-emissions"]
)
```

#### ValidationResult

Comprehensive validation response with errors and warnings.

```python
{
    "is_valid": True,
    "errors": [],  # List of validation errors
    "warnings": ["Consider adding more evidence links"]
}
```

#### QueryParams

Flexible filtering and pagination for idea queries.

```python
from pipeline.models.idea import QueryParams, IdeaStatus

params = QueryParams(
    status_filter=[IdeaStatus.VALIDATED, IdeaStatus.RESEARCHING],
    category_filter=["ai_ml", "saas"],
    search_text="machine learning",
    limit=50,
    offset=0,
    sort_by="created_at",
    sort_desc=True
)
```

---

## CLI Interface

The CLI provides comprehensive idea management through rich console commands.

### Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export DB_HOST=localhost
export DB_NAME=startup_studio
export EMBEDDING_API_KEY=your_openai_key
export SECRET_KEY=your_secret_key

# Initialize CLI
python -m pipeline.cli.ingestion_cli --help
```

### Core Commands

#### Create Ideas

Create new startup ideas with comprehensive validation.

```bash
# Interactive creation
python -m pipeline.cli.ingestion_cli create \
    --title "AI Code Assistant" \
    --description "Intelligent code completion and review system..." \
    --category ai_ml \
    --problem "Manual code review is time-consuming" \
    --solution "AI provides instant, accurate feedback" \
    --market "Software development teams" \
    --evidence "https://research.example.com/code-review-efficiency"

# Batch creation with JSON output
python -m pipeline.cli.ingestion_cli create \
    --title "Sustainable Delivery Platform" \
    --description "Carbon-neutral food delivery service..." \
    --category consumer \
    --output json \
    --force  # Skip duplicate detection
```

**Response:**
```json
{
    "idea_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "created",
    "warnings": ["Consider adding more evidence links"]
}
```

#### List and Query Ideas

Powerful filtering and search capabilities.

```bash
# List all ideas
python -m pipeline.cli.ingestion_cli list

# Filter by status and category
python -m pipeline.cli.ingestion_cli list \
    --status validated \
    --category ai_ml \
    --limit 10

# Search with text matching
python -m pipeline.cli.ingestion_cli list \
    --search "machine learning" \
    --sort title \
    --output json

# Advanced filtering
python -m pipeline.cli.ingestion_cli list \
    --stage research \
    --desc \
    --limit 25
```

#### Detailed Idea Inspection

```bash
# Show full idea details
python -m pipeline.cli.ingestion_cli show 123e4567-e89b-12d3-a456-426614174000

# JSON output for programmatic use
python -m pipeline.cli.ingestion_cli show \
    123e4567-e89b-12d3-a456-426614174000 \
    --output json
```

#### Update Existing Ideas

```bash
# Update specific fields
python -m pipeline.cli.ingestion_cli update \
    123e4567-e89b-12d3-a456-426614174000 \
    --title "Enhanced AI Code Assistant" \
    --description "Updated description with new features..." \
    --category saas

# Update market analysis
python -m pipeline.cli.ingestion_cli update \
    123e4567-e89b-12d3-a456-426614174000 \
    --market "Enterprise software teams and individual developers"
```

#### Pipeline Stage Management

```bash
# Advance to next stage
python -m pipeline.cli.ingestion_cli advance \
    123e4567-e89b-12d3-a456-426614174000 \
    research

# Available stages: ideate, research, deck, investors, mvp, smoke_test, complete
python -m pipeline.cli.ingestion_cli advance \
    123e4567-e89b-12d3-a456-426614174000 \
    deck_generation
```

#### Similarity Detection

```bash
# Find similar ideas
python -m pipeline.cli.ingestion_cli similar \
    123e4567-e89b-12d3-a456-426614174000 \
    --limit 5
```

### CLI Configuration and Health

```bash
# Show system configuration
python -m pipeline.cli.ingestion_cli config

# Check system health
python -m pipeline.cli.ingestion_cli health
```

### Output Formats

All commands support multiple output formats:

- **Table** (default): Rich console tables with color coding
- **JSON**: Machine-readable structured output for automation

---

## Service Interfaces

### IdeaManager Service

Core service for idea lifecycle management.

```python
from pipeline.ingestion.idea_manager import create_idea_manager

# Initialize manager
manager = await create_idea_manager()

# Create new idea
idea_id, warnings = await manager.create_idea(
    raw_data={
        "title": "Blockchain Supply Chain Tracker",
        "description": "Transparent supply chain using blockchain...",
        "category": "blockchain"
    },
    force_create=False,
    user_id="user_123"
)

# Retrieve idea
idea = await manager.get_idea(idea_id)

# List with filters
ideas = await manager.list_ideas(QueryParams(
    status_filter=[IdeaStatus.VALIDATED],
    limit=20
))

# Update idea
success = await manager.update_idea(
    idea_id=idea_id,
    updates={"target_market": "Supply chain managers"},
    user_id="user_123"
)

# Advance pipeline stage
success = await manager.advance_stage(
    idea_id=idea_id,
    next_stage=PipelineStage.RESEARCH,
    user_id="user_123"
)

# Find similar ideas
similar = await manager.get_similar_ideas(idea_id, limit=5)
```

### WorkflowOrchestrator Service

LangGraph-based pipeline orchestration for end-to-end processing.

```python
from pipeline.services.workflow_orchestrator import get_workflow_orchestrator

# Initialize orchestrator
orchestrator = get_workflow_orchestrator()

# Execute complete workflow
final_state = await orchestrator.execute_workflow(
    idea_id="123e4567-e89b-12d3-a456-426614174000",
    idea_data={
        "title": "AI Fitness Coach",
        "description": "Personalized AI workout and nutrition guidance...",
        "category": "healthtech"
    }
)

# Resume interrupted workflow
resumed_state = await orchestrator.resume_workflow(
    idea_id="123e4567-e89b-12d3-a456-426614174000"
)
```

**Workflow State Structure:**
```python
{
    "idea_id": "uuid-string",
    "current_stage": "research",
    "progress": 0.3,
    "started_at": "2024-01-15T10:30:00Z",
    "idea_data": {...},
    "research_data": {...},
    "deck_data": {...},
    "investor_data": {...},
    "smoke_test_data": {...},
    "mvp_data": {...},
    "quality_gates": {
        "research": "passed",
        "deck_generation": "pending"
    },
    "errors": [],
    "retry_count": 0,
    "costs_tracked": {
        "openai_api": 2.45,
        "google_ads": 15.00
    },
    "metadata": {}
}
```

### BudgetSentinel Service

Real-time cost tracking and budget enforcement.

```python
from pipeline.services.budget_sentinel import get_budget_sentinel, BudgetCategory

sentinel = get_budget_sentinel()

# Track operation with budget limits
async with sentinel.track_operation(
    service="openai_api",
    operation="embedding_generation",
    category=BudgetCategory.OPENAI,
    estimated_cost=1.50
):
    # Perform operation
    embeddings = await generate_embeddings(text)

# Check budget status
status = await sentinel.get_budget_status()
print(f"OpenAI budget: ${status['openai']['used']:.2f} / ${status['openai']['limit']:.2f}")
```

---

## Authentication and Authorization

### Environment-Based Configuration

The system uses environment variables for secure credential management:

```bash
# Required for production
export SECRET_KEY="your-secret-key-here"
export DB_PASSWORD="database-password"
export EMBEDDING_API_KEY="openai-api-key"

# Optional security settings
export ALLOWED_ORIGINS="localhost,yourdomain.com"
export ENABLE_PROFANITY_FILTER=true
export ENABLE_SPAM_DETECTION=true
```

### Security Features

**Input Validation:**
- HTML sanitization to prevent XSS attacks
- Content length limits (titles: 10-200 chars, descriptions: 10-5000 chars)
- Profanity filtering and spam detection
- URL validation for evidence links

**Access Control:**
- Environment-specific configuration validation
- Production requires SECRET_KEY and secure credentials
- Allowed origins configuration for CORS

**Data Protection:**
- Database passwords masked in logs
- Secure connection URL generation
- Correlation ID tracking for audit trails

### User Context

User identification is currently passed through the `user_id` parameter in service calls:

```python
# Create idea with user context
idea_id, warnings = await manager.create_idea(
    raw_data=idea_data,
    user_id="authenticated_user_123"
)

# Update with user tracking
success = await manager.update_idea(
    idea_id=idea_id,
    updates=updates,
    user_id="authenticated_user_123"
)
```

---

## Rate Limiting and Usage Constraints

### Content Limits

**Idea Fields:**
- Title: 10-200 characters
- Description: 10-5000 characters
- Problem statement: ≤1000 characters
- Solution description: ≤1000 characters
- Target market: ≤500 characters
- Evidence links: Valid HTTP/HTTPS URLs only

### Rate Limiting

**Per User Limits:**
- Ideas per hour: 10 (configurable via `MAX_IDEAS_PER_HOUR`)
- Ideas per day: 50 (configurable via `MAX_IDEAS_PER_DAY`)

**Query Limits:**
- List operations: 1-100 results per request (default: 20)
- Search text: ≤200 characters
- Similarity searches: ≤5 results recommended

### Budget Constraints

**Total Cycle Budget: $62.00**
- OpenAI API: $10.00 (16%)
- Google Ads: $45.00 (73%)
- Infrastructure: $5.00 (8%)
- Emergency: $2.00 (3%)

**Budget Thresholds:**
- Warning: 80% of category budget
- Critical: 95% of category budget
- Emergency shutdown: 100% of category budget

**Cost Tracking:**
- Real-time monitoring with 60-second intervals
- Automatic alerts at threshold breaches
- Emergency shutdown on budget exhaustion

---

## Error Codes and Responses

### Validation Errors

**ValidationError**: Input data validation failures
```json
{
    "error": "ValidationError",
    "message": "Title must be between 10 and 200 characters",
    "field": "title",
    "provided_value": "AI",
    "constraints": {
        "min_length": 10,
        "max_length": 200
    }
}
```

**DuplicateIdeaError**: Similar ideas detected
```json
{
    "error": "DuplicateIdeaError",
    "message": "Similar ideas found",
    "similar_ideas": [
        {
            "id": "uuid-1",
            "title": "Similar Title",
            "similarity_score": 0.85
        }
    ],
    "threshold": 0.8
}
```

### System Errors

**StorageError**: Database operation failures
```json
{
    "error": "StorageError",
    "message": "Failed to save idea to database",
    "operation": "create_idea",
    "details": "Connection timeout after 30 seconds"
}
```

**BudgetExceededException**: Cost limits reached
```json
{
    "error": "BudgetExceededException",
    "message": "Operation would exceed OpenAI budget",
    "category": "openai",
    "current_usage": 9.85,
    "limit": 10.00,
    "requested_cost": 0.50
}
```

### CLI Error Handling

CLI commands return appropriate exit codes:
- `0`: Success
- `1`: General error (validation, not found, etc.)
- `2`: System error (database, configuration, etc.)

---

## Integration Patterns

### Async/Await Pattern

All service interfaces use async/await for non-blocking operations:

```python
import asyncio

async def main():
    manager = await create_idea_manager()
    
    # Concurrent operations
    tasks = [
        manager.create_idea(data1, user_id="user1"),
        manager.create_idea(data2, user_id="user2"),
        manager.create_idea(data3, user_id="user3")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for idea_id, warnings in results:
        print(f"Created: {idea_id}, Warnings: {len(warnings)}")

# Run async main
asyncio.run(main())
```

### Error Handling Pattern

```python
from pipeline.ingestion.idea_manager import (
    ValidationError, DuplicateIdeaError, StorageError
)

async def safe_idea_creation(idea_data, user_id):
    try:
        idea_id, warnings = await manager.create_idea(
            raw_data=idea_data,
            user_id=user_id
        )
        return {"success": True, "idea_id": idea_id, "warnings": warnings}
        
    except ValidationError as e:
        return {"success": False, "error": "validation", "message": str(e)}
        
    except DuplicateIdeaError as e:
        return {"success": False, "error": "duplicate", "similar": e.similar_ideas}
        
    except StorageError as e:
        return {"success": False, "error": "storage", "message": str(e)}
        
    except Exception as e:
        return {"success": False, "error": "unknown", "message": str(e)}
```

### Batch Processing Pattern

```python
async def batch_process_ideas(idea_list, batch_size=10):
    """Process multiple ideas in controlled batches."""
    results = []
    
    for i in range(0, len(idea_list), batch_size):
        batch = idea_list[i:i + batch_size]
        batch_tasks = []
        
        for idea_data in batch:
            task = safe_idea_creation(idea_data, "batch_user")
            batch_tasks.append(task)
        
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Rate limiting between batches
        await asyncio.sleep(1)
    
    return results
```

### Configuration Management Pattern

```python
from pipeline.config.settings import get_settings

def configure_for_environment():
    """Environment-specific configuration setup."""
    settings = get_settings()
    
    if settings.is_production():
        # Production-specific setup
        assert settings.secret_key, "SECRET_KEY required in production"
        log_level = "WARNING"
        enable_debug = False
    else:
        # Development-specific setup
        log_level = "DEBUG"
        enable_debug = True
    
    return {
        "log_level": log_level,
        "debug": enable_debug,
        "database_url": settings.database.get_safe_connection_url()
    }
```

---

## Configuration Reference

### Environment Variables

**Database Configuration:**
```bash
DB_HOST=localhost              # Database host (default: localhost)
DB_PORT=5432                  # Database port (default: 5432)
DB_NAME=startup_studio        # Database name (default: startup_studio)
DB_USER=postgres              # Database user (default: postgres)
DB_PASSWORD=secret            # Database password (required in production)
DB_MIN_CONNECTIONS=1          # Connection pool minimum (default: 1)
DB_MAX_CONNECTIONS=20         # Connection pool maximum (default: 20)
DB_TIMEOUT=30                 # Connection timeout seconds (default: 30)
VECTOR_DIMENSIONS=1536        # Embedding vector size (default: 1536)
ENABLE_VECTOR_SEARCH=true     # Enable similarity search (default: true)
```

**Validation Configuration:**
```bash
MIN_TITLE_LENGTH=10           # Minimum title length (default: 10)
MAX_TITLE_LENGTH=200          # Maximum title length (default: 200)
MIN_DESCRIPTION_LENGTH=10     # Minimum description length (default: 10)
MAX_DESCRIPTION_LENGTH=5000   # Maximum description length (default: 5000)
SIMILARITY_THRESHOLD=0.8      # Duplicate detection threshold (default: 0.8)
TITLE_FUZZY_THRESHOLD=0.7     # Title similarity threshold (default: 0.7)
MAX_IDEAS_PER_HOUR=10         # Rate limit per hour (default: 10)
MAX_IDEAS_PER_DAY=50          # Rate limit per day (default: 50)
ENABLE_PROFANITY_FILTER=true  # Content filtering (default: true)
ENABLE_SPAM_DETECTION=true    # Spam detection (default: true)
```

**AI/Embedding Configuration:**
```bash
EMBEDDING_PROVIDER=openai     # Provider: openai, huggingface (default: openai)
EMBEDDING_API_KEY=sk-...      # API key (required for OpenAI)
EMBEDDING_MODEL=text-embedding-ada-002  # Model name
ENABLE_EMBEDDING_CACHE=true   # Cache embeddings (default: true)
EMBEDDING_CACHE_TTL=86400     # Cache TTL seconds (default: 24 hours)
EMBEDDING_BATCH_SIZE=10       # Batch processing size (default: 10)
EMBEDDING_RETRY_ATTEMPTS=3    # Retry failed requests (default: 3)
```

**Budget Configuration:**
```bash
TOTAL_CYCLE_BUDGET=62.00      # Total budget per cycle (default: $62.00)
OPENAI_BUDGET=10.00           # OpenAI allocation (default: $10.00)
GOOGLE_ADS_BUDGET=45.00       # Google Ads allocation (default: $45.00)
INFRASTRUCTURE_BUDGET=5.00    # Infrastructure allocation (default: $5.00)
BUDGET_WARNING_THRESHOLD=0.80 # Warning at 80% (default: 0.80)
BUDGET_CRITICAL_THRESHOLD=0.95 # Critical at 95% (default: 0.95)
ENABLE_BUDGET_ALERTS=true     # Enable alerts (default: true)
ENABLE_EMERGENCY_SHUTDOWN=true # Emergency stop (default: true)
```

**Logging Configuration:**
```bash
LOG_LEVEL=INFO                # Log level: DEBUG, INFO, WARNING, ERROR
DB_LOG_LEVEL=WARNING          # Database log level (default: WARNING)
ENABLE_JSON_LOGGING=true      # JSON format logs (default: true)
ENABLE_CORRELATION_IDS=true   # Request tracking (default: true)
LOG_FILE=/var/log/app.log     # Log file path (optional)
ENABLE_METRICS=true           # Enable metrics collection (default: true)
```

**Application Configuration:**
```bash
ENVIRONMENT=production        # Environment: development, testing, staging, production
DEBUG_MODE=false              # Debug mode (default: false)
SECRET_KEY=your-secret-key    # Application secret (required in production)
ALLOWED_ORIGINS=localhost,yourdomain.com  # CORS origins (comma-separated)
APP_NAME=agentic-startup-studio # Application name
APP_VERSION=1.0.0             # Application version
```

### Configuration Validation

The system validates configuration at startup:

```python
from pipeline.config.settings import validate_required_env_vars

# Validate required environment variables
try:
    validate_required_env_vars()
    print("Configuration validation passed")
except ValueError as e:
    print(f"Configuration error: {e}")
    exit(1)
```

For production deployments, ensure all required environment variables are set and valid.