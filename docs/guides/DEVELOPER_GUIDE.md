# Developer Guide - Agentic Startup Studio

## Architecture Overview

The Agentic Startup Studio is built on a modern, event-driven microservices architecture with comprehensive SDLC practices.

### Core Components
- **Pipeline Layer**: Async processing with AI agents
- **API Layer**: FastAPI with JWT authentication
- **Storage Layer**: PostgreSQL with pgvector for embeddings
- **Monitoring Layer**: OpenTelemetry + Prometheus + Grafana
- **Testing Layer**: pytest with 90%+ coverage

## Development Environment

### Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/agentic-startup-studio.git
cd agentic-startup-studio

# Setup development environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Development Tools
- **Code Formatting**: Black + isort
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: MyPy with strict configuration
- **Security**: Bandit for security scanning
- **Testing**: pytest with coverage and async support

## Code Style Guidelines

### Python Standards
- **PEP 8**: Enforced via Black (line length: 88)
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all modules, classes, and functions
- **Import Organization**: isort with project-specific configuration

### Code Quality Standards
```python
# Example of well-structured code
from typing import Optional, Dict, Any
import logging

from pydantic import BaseModel
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class IdeaRequest(BaseModel):
    """Request model for idea validation."""
    
    idea: str
    budget: Optional[float] = 50.0
    evidence_depth: str = "standard"
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True

async def validate_idea(request: IdeaRequest) -> Dict[str, Any]:
    """Validate a startup idea using AI agents.
    
    Args:
        request: Idea validation request
        
    Returns:
        Validation results with evidence and scoring
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Implementation here
        logger.info(f"Validating idea: {request.idea[:50]}...")
        return {"status": "validated"}
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail="Validation failed")
```

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
├── performance/   # Load and performance tests
├── fixtures/      # Test data and utilities
└── conftest.py    # Shared pytest configuration
```

### Writing Tests
```python
# Example test structure
import pytest
from unittest.mock import AsyncMock, patch

from pipeline.services.idea_validator import IdeaValidator
from pipeline.models.idea import IdeaRequest

class TestIdeaValidator:
    """Test suite for IdeaValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return IdeaValidator()
    
    @pytest.mark.asyncio
    async def test_validate_idea_success(self, validator):
        """Test successful idea validation."""
        request = IdeaRequest(idea="AI-powered task management")
        
        with patch('pipeline.services.ai_client.OpenAIClient') as mock_ai:
            mock_ai.return_value.generate.return_value = {"score": 0.85}
            
            result = await validator.validate(request)
            
            assert result["score"] >= 0.8
            assert "evidence" in result
            mock_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_idea_budget_exceeded(self, validator):
        """Test validation with budget constraints."""
        request = IdeaRequest(idea="Expensive validation", budget=1.0)
        
        with pytest.raises(BudgetExceededException):
            await validator.validate(request)
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=pipeline --cov=core --cov-report=html

# Run performance tests
pytest tests/performance/ -v
```

## API Development

### FastAPI Patterns
```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer

from pipeline.auth import verify_token
from pipeline.models import IdeaRequest, IdeaResponse

router = APIRouter(prefix="/api/v1", tags=["ideas"])
security = HTTPBearer()

@router.post("/ideas/validate", response_model=IdeaResponse)
async def validate_idea(
    request: IdeaRequest,
    token: str = Depends(security),
    current_user: dict = Depends(verify_token)
) -> IdeaResponse:
    """Validate a startup idea."""
    # Validation logic here
    pass
```

### Error Handling
```python
from pipeline.exceptions import ValidationError, BudgetExceededError

@router.post("/ideas/validate")
async def validate_idea(request: IdeaRequest):
    try:
        return await idea_service.validate(request)
    except BudgetExceededError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Database Development

### Model Patterns
```python
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class IdeaBase(SQLModel):
    """Base idea model."""
    title: str = Field(max_length=200)
    description: str
    budget: float = Field(default=50.0, ge=0, le=1000)

class Idea(IdeaBase, table=True):
    """Idea database model."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Vector embedding for similarity search
    embedding: Optional[str] = Field(default=None)
```

### Repository Patterns
```python
from typing import List, Optional
from sqlmodel import Session, select

class IdeaRepository:
    """Repository for idea data access."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def create(self, idea: IdeaBase) -> Idea:
        """Create new idea."""
        db_idea = Idea.from_orm(idea)
        self.session.add(db_idea)
        await self.session.commit()
        await self.session.refresh(db_idea)
        return db_idea
    
    async def find_similar(self, embedding: str, limit: int = 5) -> List[Idea]:
        """Find similar ideas using vector search."""
        statement = select(Idea).where(
            Idea.embedding.op("<->")(embedding) < 0.3
        ).limit(limit)
        
        results = await self.session.exec(statement)
        return results.all()
```

## Monitoring and Observability

### Logging
```python
import logging
from opentelemetry import trace

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

async def process_idea(idea: str):
    """Process idea with tracing and logging."""
    with tracer.start_as_current_span("process_idea") as span:
        span.set_attribute("idea.length", len(idea))
        
        logger.info(f"Processing idea: {idea[:50]}...")
        
        try:
            result = await ai_service.analyze(idea)
            span.set_attribute("result.score", result.score)
            logger.info(f"Idea processed successfully, score: {result.score}")
            return result
        except Exception as e:
            span.record_exception(e)
            logger.error(f"Failed to process idea: {e}")
            raise
```

### Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
idea_validations = Counter(
    'idea_validations_total',
    'Total number of idea validations',
    ['status', 'user_type']
)

validation_duration = Histogram(
    'idea_validation_duration_seconds',
    'Time spent validating ideas',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

active_validations = Gauge(
    'active_validations',
    'Number of currently active validations'
)

# Usage in code
@validation_duration.time()
async def validate_idea(request: IdeaRequest):
    active_validations.inc()
    try:
        result = await idea_service.validate(request)
        idea_validations.labels(status='success', user_type='free').inc()
        return result
    except Exception:
        idea_validations.labels(status='error', user_type='free').inc()
        raise
    finally:
        active_validations.dec()
```

## Performance Guidelines

### Database Optimization
- Use connection pooling for database connections
- Implement proper indexing for vector searches
- Use async/await for all database operations
- Implement query result caching where appropriate

### API Optimization
- Use async handlers for all endpoints
- Implement request/response caching
- Use connection pooling for external APIs
- Implement circuit breakers for external dependencies

### Memory Management
- Use generators for large data processing
- Implement proper cleanup for long-running tasks
- Monitor memory usage in production
- Use streaming for large responses

## Security Guidelines

### Authentication & Authorization
- All API endpoints require JWT authentication
- Implement role-based access control
- Use secure password hashing (bcrypt)
- Implement rate limiting per user/IP

### Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement proper input validation
- Sanitize all user inputs

### Dependency Security
- Regular security scanning with bandit
- Automated dependency updates with renovate
- Pin dependency versions in production
- Regular security audits

## Deployment Guidelines

### Docker Best Practices
- Use multi-stage builds for optimization
- Run containers as non-root user
- Implement health checks
- Use specific base image versions

### Environment Configuration
- Use environment variables for configuration
- Never commit secrets to repository
- Use secret management systems in production
- Implement configuration validation

### Monitoring Setup
- Configure comprehensive logging
- Set up health check endpoints
- Implement performance monitoring
- Configure alerting rules

## Contributing Guidelines

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite and linting
4. Update documentation as needed
5. Create PR with detailed description
6. Address review feedback
7. Merge after approval

### Code Review Checklist
- [ ] Tests cover new functionality
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance impact assessed
- [ ] Backward compatibility maintained
- [ ] Error handling implemented
- [ ] Logging added where appropriate

## Resources

### Documentation
- [Architecture Overview](../ARCHITECTURE.md)
- [API Documentation](../api-documentation-v2.md)
- [Testing Framework](../testing-framework-architecture.md)
- [Deployment Guide](../deployment/README.md)

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [PostgreSQL Vector Extension](https://github.com/pgvector/pgvector)