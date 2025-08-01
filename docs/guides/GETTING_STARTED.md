# Getting Started with Agentic Startup Studio

## Quick Start Guide

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- PostgreSQL (if running locally)
- OpenAI API key

### 1. Environment Setup

#### Clone and Install
```bash
git clone https://github.com/danieleschmidt/agentic-startup-studio.git
cd agentic-startup-studio

# Install dependencies
pip install -e ".[dev]"

# Or use UV for faster installation
uv pip install -e ".[dev]"
```

#### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys and configuration
vi .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_AI_API_KEY`: Google AI API key (optional)
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET_KEY`: Secret for JWT token generation

### 2. Development Environment

#### Using DevContainer (Recommended)
```bash
# Open in VS Code with DevContainer
code .
# Command Palette: "Dev Containers: Reopen in Container"
```

#### Local Development
```bash
# Start services
docker-compose up -d

# Run database migrations
python -m pipeline.cli migrate

# Start the API server
python -m pipeline.api.gateway

# In another terminal, start the health server
python -m scripts.run_health_checks
```

### 3. CLI Usage

#### Basic Commands
```bash
# Validate a startup idea
studio validate "AI-powered task management app"

# Run health checks
studio-health

# Start API server
studio-api
```

#### Advanced Usage
```bash
# Custom validation with budget control
studio validate "SaaS productivity tool" --budget 50 --evidence-depth high

# Batch processing
studio batch-validate ideas.json
```

### 4. Testing

#### Run Test Suite
```bash
# Full test suite
pytest

# Quick tests only
pytest -m "not slow"

# With coverage
pytest --cov=pipeline --cov=core
```

#### Performance Testing
```bash
# Load testing
k6 run tests/performance/load-test.js

# Spike testing  
k6 run tests/performance/spike-test.js
```

### 5. Monitoring

#### Local Monitoring Stack
```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

#### Health Checks
```bash
# System health
curl http://localhost:8000/health

# Detailed health status
curl http://localhost:8000/health/detailed
```

### 6. API Usage

#### Authentication
```bash
# Get access token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

#### Validate Ideas
```bash
# Submit idea for validation
curl -X POST http://localhost:8000/api/v1/ideas/validate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"idea": "AI-powered personal assistant", "budget": 50}'
```

### 7. Troubleshooting

#### Common Issues

**Database Connection Errors**
```bash
# Check database status
docker-compose ps postgres

# View logs
docker-compose logs postgres
```

**API Key Issues**
```bash
# Verify environment variables
echo $OPENAI_API_KEY

# Test API connectivity
python -c "import openai; print(openai.api_key[:10] + '...')"
```

**Performance Issues**
```bash
# Check system resources
docker stats

# Review performance metrics
curl http://localhost:8000/metrics
```

#### Getting Help
- 📖 **Documentation**: `/docs` directory
- 🐛 **Issues**: [GitHub Issues](https://github.com/danieleschmidt/agentic-startup-studio/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/danieleschmidt/agentic-startup-studio/discussions)
- 📧 **Support**: dev@terragonlabs.com

### 8. Development Workflow

#### Making Changes
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
pytest
ruff check
mypy pipeline/

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/your-feature
```

#### Code Quality
```bash
# Format code
black .
isort .

# Lint code
ruff check --fix

# Type checking
mypy pipeline/ core/

# Security scanning
bandit -r pipeline/ core/
```

### Next Steps
- 📚 Review [Architecture Documentation](../ARCHITECTURE.md)
- 🧪 Explore [Testing Framework](../testing-framework-architecture.md)
- 🚀 Deploy to [Production](../deployment/README.md)
- 📊 Setup [Monitoring](../monitoring/README.md)