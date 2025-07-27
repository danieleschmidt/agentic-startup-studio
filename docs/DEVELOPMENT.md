# Developer Onboarding Guide

Welcome to the Agentic Startup Studio development team! This guide will help you get up and running quickly with our development environment and processes.

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - Our primary development language
- **Docker & Docker Compose** - For containerized development
- **Git** - Version control
- **VS Code** (recommended) - IDE with devcontainer support
- **PostgreSQL 14+** - Database (or use Docker)
- **Node.js 18+** - For frontend tooling and testing

### Option 1: Devcontainer (Recommended)

The fastest way to get started is using our devcontainer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/terragonlabs/agentic-startup-studio.git
   cd agentic-startup-studio
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Reopen in Container:**
   - VS Code will prompt to "Reopen in Container"
   - Or use Command Palette: `Remote-Containers: Reopen in Container`

4. **Wait for setup:**
   - The devcontainer will automatically install dependencies
   - Health checks will run to verify the setup

### Option 2: Local Development

If you prefer local development:

1. **Set up Python environment:**
   ```bash
   # Using UV (recommended)
   python uv-setup.py
   
   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start development services:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **Run the application:**
   ```bash
   python scripts/serve_api.py --port 8000 --reload
   ```

### Verification

Verify your setup by running:

```bash
# Health check
curl http://localhost:8000/health

# Run tests
pytest

# Check code quality
ruff check .
mypy .
```

## 📁 Project Structure

```
agentic-startup-studio/
├── .devcontainer/          # Development container configuration
├── .github/                # GitHub workflows and templates
├── .vscode/                # VS Code settings and extensions
├── docs/                   # Documentation
│   ├── api/               # API documentation
│   ├── guides/            # User and developer guides
│   ├── runbooks/          # Operational runbooks
│   └── specs/             # Technical specifications
├── pipeline/               # Core application code
│   ├── api/               # API layer
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration management
│   ├── core/              # Core business logic
│   ├── ingestion/         # Data ingestion
│   ├── models/            # Data models
│   ├── services/          # Business services
│   └── storage/           # Data persistence
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── e2e/               # End-to-end tests
│   └── performance/       # Performance tests
├── scripts/                # Utility scripts
├── monitoring/             # Monitoring configuration
├── tools/                  # External tool integrations
└── requirements.txt        # Python dependencies
```

## 🛠️ Development Workflow

### 1. Feature Development

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop your feature:**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   
   # Check coverage
   pytest --cov=pipeline --cov-report=html
   ```

4. **Code quality checks:**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   ruff check .
   
   # Type checking
   mypy .
   
   # Security scan
   bandit -r .
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a pull request:**
   - Use our PR template
   - Ensure all CI checks pass
   - Request review from team members

### 2. Code Style Guidelines

We follow these code style conventions:

#### Python Style
- **Formatter:** Black (line length 88)
- **Import sorting:** isort with Black profile
- **Linting:** Ruff
- **Type hints:** Required for all public functions
- **Docstrings:** Google style

Example:
```python
from typing import List, Optional

from pydantic import BaseModel

from pipeline.models.idea import Idea


class IdeaService:
    """Service for managing startup ideas.
    
    This service provides methods for creating, updating, and retrieving
    startup ideas from the database.
    """
    
    def __init__(self, repository: IdeaRepository) -> None:
        """Initialize the service with a repository.
        
        Args:
            repository: The idea repository instance.
        """
        self.repository = repository
    
    async def create_idea(
        self, 
        title: str, 
        description: str,
        category: Optional[str] = None
    ) -> Idea:
        """Create a new startup idea.
        
        Args:
            title: The idea title.
            description: The idea description.
            category: Optional category classification.
            
        Returns:
            The created idea instance.
            
        Raises:
            ValidationError: If the input data is invalid.
        """
        # Implementation here
        pass
```

#### API Design
- **REST principles:** Use appropriate HTTP methods
- **Status codes:** Follow HTTP standards
- **Error handling:** Consistent error responses
- **Validation:** Use Pydantic models
- **Documentation:** OpenAPI/Swagger specs

#### Database
- **Migrations:** Use Alembic for schema changes
- **Queries:** Use SQLAlchemy ORM
- **Indexing:** Proper indexing for performance
- **Transactions:** Appropriate transaction boundaries

### 3. Testing Guidelines

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch

from pipeline.services.idea_service import IdeaService
from pipeline.models.idea import Idea


class TestIdeaService:
    """Test suite for IdeaService."""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing."""
        return Mock()
    
    @pytest.fixture
    def service(self, mock_repository):
        """Service instance for testing."""
        return IdeaService(mock_repository)
    
    async def test_create_idea_success(self, service, mock_repository):
        """Test successful idea creation."""
        # Arrange
        mock_repository.create.return_value = Idea(
            id="test-id",
            title="Test Idea",
            description="Test Description"
        )
        
        # Act
        result = await service.create_idea(
            title="Test Idea",
            description="Test Description"
        )
        
        # Assert
        assert result.title == "Test Idea"
        assert result.description == "Test Description"
        mock_repository.create.assert_called_once()
    
    async def test_create_idea_validation_error(self, service):
        """Test idea creation with invalid data."""
        with pytest.raises(ValidationError):
            await service.create_idea(title="", description="")
```

#### Test Categories
- **Unit tests:** Test individual functions/methods
- **Integration tests:** Test component interactions
- **E2E tests:** Test complete user workflows
- **Performance tests:** Test system performance
- **Security tests:** Test security controls

## 🔧 Tools and Commands

### Development Commands

```bash
# Start development server with hot reload
python scripts/serve_api.py --reload

# Run specific pipeline stages
python -m pipeline.cli.ingestion_cli create --title "Test Idea"

# Database operations
python scripts/setup_production_secrets.py --dev-mode

# Health checks
python scripts/run_health_checks.py

# Performance testing
k6 run tests/performance/load-test.js

# Security scanning
python scripts/security_scan.py --scan-type all

# Generate API documentation
python scripts/generate_api_docs.py
```

### Debugging

#### Using VS Code Debugger
1. Set breakpoints in your code
2. Use F5 or Run → Start Debugging
3. Select "Python: FastAPI" configuration

#### Using IPython/IPdb
```python
import ipdb; ipdb.set_trace()  # Set breakpoint
```

#### Logging
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Debug information")
logger.error("Error occurred", exc_info=True)
```

### Database Management

```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Connect to database
psql -h localhost -U studio -d studio
```

## 🚀 Deployment

### Local Deployment
```bash
# Build container
docker build -t agentic-studio .

# Run container
docker run -p 8000:8000 agentic-studio
```

### Staging Deployment
```bash
# Deploy to staging
git push origin staging

# Check deployment status
kubectl get pods -n staging
```

### Production Deployment
```bash
# Create release
git tag v1.2.3
git push origin v1.2.3

# Monitor deployment
kubectl logs -f deployment/agentic-studio -n production
```

## 📚 Additional Resources

### Documentation
- [API Documentation](./api-documentation.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Security Guidelines](../SECURITY.md)
- [Operations Manual](./operations-manual.md)

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Team Resources
- **Slack Channels:**
  - `#development` - General development discussion
  - `#platform` - Platform and infrastructure
  - `#security` - Security-related topics
  - `#oncall` - On-call and incidents

- **Meetings:**
  - Daily Standup: 9:00 AM PST
  - Sprint Planning: Every other Monday
  - Retrospective: Every other Friday
  - Architecture Review: Thursdays 2:00 PM PST

## 🆘 Getting Help

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

#### Database connection issues
```bash
# Check if PostgreSQL is running
docker-compose ps

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

#### Tests failing
```bash
# Run tests with verbose output
pytest -v --tb=long

# Run specific test
pytest tests/unit/test_specific.py::TestClass::test_method -v
```

### Who to Contact

| Issue Type | Contact | Response Time |
|------------|---------|---------------|
| Development questions | Team lead or senior dev | Same day |
| Environment issues | Platform team | 2-4 hours |
| Security concerns | Security team | Immediate |
| Urgent production issues | On-call engineer | Immediate |

### Code Review Process

1. **Self-review:** Review your own PR first
2. **Automated checks:** Ensure all CI checks pass
3. **Peer review:** Request review from 1-2 team members
4. **Address feedback:** Make requested changes
5. **Approval:** Get approval before merging
6. **Merge:** Use "Squash and merge" for clean history

## 📈 Performance Considerations

### Database Performance
- Use database indexes appropriately
- Avoid N+1 queries
- Use connection pooling
- Monitor slow queries

### API Performance
- Implement caching where appropriate
- Use async/await for I/O operations
- Implement proper pagination
- Monitor response times

### Memory Usage
- Be mindful of large data structures
- Use generators for large datasets
- Monitor memory usage in production
- Implement proper cleanup

---

Welcome to the team! If you have any questions, don't hesitate to ask in the `#development` Slack channel or reach out to your mentor.

*Last updated: 2025-07-27*