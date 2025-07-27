# Contributing to Agentic Startup Studio

Thank you for your interest in contributing to the Agentic Startup Studio! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

Before contributing, please ensure you have:

- Python 3.11 or higher
- Docker and Docker Compose
- Git
- Basic understanding of FastAPI, PostgreSQL, and AI/ML concepts

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/agentic-startup-studio.git
   cd agentic-startup-studio
   ```
3. **Set up the development environment**:
   ```bash
   # Using devcontainer (recommended)
   code .  # Open in VS Code and use "Reopen in Container"
   
   # Or local setup
   python uv-setup.py
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with appropriate values
   ```
5. **Run tests** to verify setup:
   ```bash
   pytest
   ```

## 📝 How to Contribute

### Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating a new issue
3. **Include detailed information**:
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and logs
   - Screenshots if applicable

### Suggesting Features

Feature requests are welcome! Please:

1. **Check existing feature requests** to avoid duplicates
2. **Use the feature request template**
3. **Provide detailed information**:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach
   - Any relevant examples or references

### Contributing Code

#### Types of Contributions

We welcome contributions in these areas:

- **Bug fixes** - Fixing identified issues
- **Feature development** - Implementing new functionality
- **Performance improvements** - Optimizing existing code
- **Documentation** - Improving or adding documentation
- **Testing** - Adding or improving tests
- **Security** - Enhancing security measures
- **DevOps** - Improving CI/CD and infrastructure

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make your changes**:
   - Follow our [coding standards](#coding-standards)
   - Write tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "type(scope): description"
   ```
   
   Use [conventional commits](https://www.conventionalcommits.org/):
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Use the PR template
   - Provide clear description of changes
   - Link related issues
   - Ensure all CI checks pass

## 📋 Coding Standards

### Python Code Style

We follow these standards:

- **PEP 8** compliance with line length of 88 characters
- **Black** for code formatting
- **isort** for import sorting
- **Type hints** required for all public functions
- **Docstrings** in Google style for all public functions and classes

Example:
```python
from typing import List, Optional

from pydantic import BaseModel


class ExampleService:
    """Service for demonstrating coding standards.
    
    This service shows how to properly structure code according
    to our coding standards and conventions.
    """
    
    def __init__(self, config: dict) -> None:
        """Initialize the service with configuration.
        
        Args:
            config: Configuration dictionary containing service settings.
        """
        self.config = config
    
    async def process_data(
        self, 
        data: List[dict], 
        validate: bool = True
    ) -> Optional[dict]:
        """Process the provided data according to business rules.
        
        Args:
            data: List of data items to process.
            validate: Whether to validate data before processing.
            
        Returns:
            Processed data summary or None if processing failed.
            
        Raises:
            ValidationError: If data validation fails.
            ProcessingError: If data processing encounters an error.
        """
        # Implementation here
        pass
```

### API Design Guidelines

- **RESTful principles** - Use appropriate HTTP methods and status codes
- **Consistent naming** - Use snake_case for JSON fields
- **Proper error handling** - Return meaningful error messages
- **Input validation** - Use Pydantic models for request/response validation
- **Documentation** - All endpoints must have OpenAPI documentation

### Database Guidelines

- **Migrations** - Use Alembic for all schema changes
- **Naming conventions** - Use snake_case for tables and columns
- **Indexing** - Add appropriate indexes for query performance
- **Transactions** - Use proper transaction boundaries
- **Security** - Never use string interpolation for queries

### Testing Guidelines

- **Test coverage** - Aim for >90% code coverage
- **Test categories** - Write unit, integration, and e2e tests
- **Test naming** - Use descriptive test names that explain the scenario
- **Test structure** - Follow Arrange-Act-Assert pattern
- **Mocking** - Use mocks appropriately to isolate units under test

Example test:
```python
import pytest
from unittest.mock import Mock

from pipeline.services.example_service import ExampleService


class TestExampleService:
    """Test suite for ExampleService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {"timeout": 30, "retries": 3}
    
    @pytest.fixture
    def service(self, mock_config):
        """Service instance for testing."""
        return ExampleService(mock_config)
    
    async def test_process_data_success(self, service):
        """Test successful data processing with valid input."""
        # Arrange
        test_data = [{"id": 1, "value": "test"}]
        
        # Act
        result = await service.process_data(test_data)
        
        # Assert
        assert result is not None
        assert result["status"] == "success"
    
    async def test_process_data_validation_error(self, service):
        """Test data processing with invalid input raises ValidationError."""
        # Arrange
        invalid_data = [{"invalid": "data"}]
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await service.process_data(invalid_data)
```

## 🔍 Code Review Process

### Submitting PRs

1. **Self-review** your code before requesting review
2. **Ensure CI passes** - All automated checks must pass
3. **Write clear descriptions** - Explain what and why, not just how
4. **Keep PRs focused** - One feature or fix per PR
5. **Update documentation** - Include relevant documentation updates

### Review Criteria

Reviewers will check for:

- **Functionality** - Does the code work as intended?
- **Code quality** - Is the code clean, readable, and maintainable?
- **Performance** - Are there any performance implications?
- **Security** - Does the code introduce security vulnerabilities?
- **Testing** - Are there adequate tests?
- **Documentation** - Is documentation updated appropriately?

### Responding to Reviews

- **Address all feedback** - Respond to every comment
- **Ask for clarification** - If feedback is unclear, ask questions
- **Make requested changes** - Update code based on feedback
- **Update descriptions** - Keep PR description current with changes

## 🚨 Security Guidelines

### Reporting Security Issues

- **Do not** create public GitHub issues for security vulnerabilities
- **Email** security@terragonlabs.com with details
- **Follow** our [Security Policy](SECURITY.md)

### Secure Coding Practices

- **Input validation** - Validate all user inputs
- **SQL injection prevention** - Use parameterized queries
- **Authentication** - Implement proper authentication and authorization
- **Secrets management** - Never hardcode secrets in code
- **Error handling** - Don't expose sensitive information in error messages

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=pipeline --cov-report=html

# Run performance tests
k6 run tests/performance/load-test.js
```

### Test Requirements

- **New features** must include unit tests
- **Bug fixes** must include regression tests
- **API changes** must include integration tests
- **Performance changes** should include performance tests

## 📚 Documentation

### Types of Documentation

- **Code documentation** - Docstrings and inline comments
- **API documentation** - OpenAPI/Swagger specifications
- **User guides** - How to use the system
- **Developer guides** - How to contribute and develop
- **Operational docs** - Deployment and maintenance

### Documentation Standards

- **Clear and concise** - Write for your audience
- **Up-to-date** - Keep documentation current with code changes
- **Examples** - Include practical examples
- **Searchable** - Use descriptive headings and keywords

## 🎯 Issue Triage and Labels

### Issue Labels

We use these labels to categorize issues:

- **Type labels**: `bug`, `feature`, `documentation`, `security`, `performance`
- **Priority labels**: `priority:low`, `priority:medium`, `priority:high`, `priority:critical`
- **Status labels**: `status:triaged`, `status:in-progress`, `status:blocked`
- **Area labels**: `area:api`, `area:database`, `area:ui`, `area:tests`, `area:docs`

### Issue Lifecycle

1. **New** - Issue is created
2. **Triaged** - Issue is reviewed and labeled
3. **Assigned** - Issue is assigned to contributor
4. **In Progress** - Work has started
5. **Review** - PR is submitted and under review
6. **Closed** - Issue is resolved

## 🏆 Recognition

We appreciate all contributions! Contributors will be:

- **Listed** in our README acknowledgments
- **Mentioned** in release notes for significant contributions
- **Invited** to special contributor events
- **Offered** priority support for their own projects

## 📞 Getting Help

### Communication Channels

- **GitHub Discussions** - General questions and discussions
- **GitHub Issues** - Bug reports and feature requests
- **Slack** (team members only) - Real-time collaboration
- **Email** - Direct contact for sensitive issues

### Response Times

- **Bug reports** - 2-3 business days
- **Feature requests** - 1 week
- **Security issues** - 24-48 hours
- **PR reviews** - 2-5 business days

### Office Hours

We hold virtual office hours for contributors:
- **When**: Every other Friday, 2:00-3:00 PM PST
- **Where**: Zoom link in GitHub Discussions
- **Topics**: Questions, contributions, roadmap discussions

## 📋 Contributor License Agreement

By contributing to this project, you agree that:

1. Your contributions are original work
2. You have the right to submit the contributions
3. Your contributions are licensed under the MIT License
4. You grant us the right to use, modify, and distribute your contributions

## 🗺️ Roadmap and Planning

### Current Priorities

1. **Security enhancements** - Improving security posture
2. **Performance optimization** - Faster response times
3. **API stability** - Stable v1 API release
4. **Documentation** - Comprehensive user guides

### How to Influence the Roadmap

- **Participate** in GitHub Discussions
- **Submit** well-researched feature requests
- **Contribute** to priority issues
- **Engage** with the community

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Thank you for contributing to Agentic Startup Studio! Your contributions help make AI-powered startup validation accessible to everyone.

*Last updated: 2025-07-27*