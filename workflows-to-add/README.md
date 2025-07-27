# GitHub Actions Workflows

Due to GitHub App permissions restrictions, these workflow files need to be added manually to your repository.

## How to Add These Workflows

1. **Create the workflows directory** (if it doesn't exist):
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy the workflow files**:
   ```bash
   cp workflows-to-add/ci.yml .github/workflows/
   cp workflows-to-add/cd.yml .github/workflows/
   ```

3. **Commit and push**:
   ```bash
   git add .github/workflows/
   git commit -m "feat: add comprehensive CI/CD workflows

   - Enhanced CI pipeline with parallel jobs for code quality, security, and testing
   - Comprehensive CD pipeline with blue-green deployment strategy
   - Multi-environment support (staging/production)
   - Automated security scanning and vulnerability detection
   - Build validation and performance testing
   - Automated release management and PyPI publishing
   
   🤖 Generated with [Claude Code](https://claude.ai/code)"
   git push
   ```

## Workflow Overview

### 🔄 Continuous Integration (`ci.yml`)

**Triggers**: Push to main/develop, Pull Requests, Manual dispatch

**Jobs**:
- **🔍 Code Quality**: Black, isort, Ruff, MyPy
- **🔒 Security Scan**: Bandit, Safety, secrets detection
- **🧪 Tests**: Unit tests with 90% coverage requirement (Python 3.11 & 3.12)
- **🔗 Integration Tests**: Full pipeline integration testing
- **🐳 Docker Build**: Container build and validation
- **📦 Dependency Check**: Conflict detection and security analysis
- **🏗️ Build Validation**: Package building and validation
- **⚡ Performance Tests**: Benchmark execution (main branch only)

### 🚀 Continuous Deployment (`cd.yml`)

**Triggers**: Push to main, Tags (v*), Manual dispatch

**Jobs**:
- **🏗️ Build & Publish**: Multi-arch Docker images, Python packages
- **🔒 Security Scan**: Trivy vulnerability scanning
- **🧪 Deploy to Staging**: Automated staging deployment with smoke tests
- **🏭 Deploy to Production**: Blue-green production deployment (tags only)
- **📦 Create Release**: Automated GitHub releases with changelog
- **📦 Publish to PyPI**: Package publishing to PyPI (tags only)
- **🔄 Prepare Rollback**: Automated rollback preparation on failure
- **📢 Notify**: Deployment notifications

## Features

### 🔧 Development Features
- **Multi-version testing** (Python 3.11, 3.12)
- **Parallel job execution** for faster feedback
- **Comprehensive coverage reporting** with Codecov integration
- **Dependency caching** for faster builds
- **Matrix testing** across environments

### 🔒 Security Features
- **Multi-layer security scanning** (Bandit, Safety, Trivy)
- **Secrets detection** and validation
- **Container vulnerability scanning**
- **SARIF reporting** for security findings
- **Dependency vulnerability tracking**

### 🚀 Deployment Features
- **Blue-green deployment** strategy
- **Multi-environment** support (staging/production)
- **Automated rollback** capabilities
- **Smoke testing** after deployments
- **Container registry** integration (GitHub Container Registry)
- **Release management** with automated changelog

### 📊 Quality Gates
- **90% test coverage** requirement
- **Code formatting** enforcement
- **Type checking** with MyPy
- **Security vulnerability** scanning
- **Performance benchmarking**

## Environment Setup

### Required Secrets

Add these secrets to your GitHub repository settings:

```bash
# For PyPI publishing (optional)
PYPI_API_TOKEN=<your-pypi-token>

# For notifications (optional)
SLACK_WEBHOOK=<your-slack-webhook-url>

# For additional integrations (optional)
CODECOV_TOKEN=<your-codecov-token>
```

### Environment Protection Rules

Configure environment protection rules in GitHub:

1. **Staging Environment**:
   - No protection rules (auto-deploy from main)
   - Environment URL: `https://staging.agentic-startup-studio.com`

2. **Production Environment**:
   - Required reviewers (recommended)
   - Deployment branches: Tags only (`v*`)
   - Environment URL: `https://agentic-startup-studio.com`

3. **PyPI Environment**:
   - Required reviewers
   - Deployment branches: Tags only (`v*`)

## Customization

### Deployment Commands

Update the deployment commands in both workflows to match your infrastructure:

```yaml
# Example Kubernetes deployment
kubectl set image deployment/app app=image:tag
kubectl rollout status deployment/app

# Example Cloud Run deployment
gcloud run deploy app --image image:tag --region us-central1

# Example Docker Compose deployment
export IMAGE_TAG=tag
docker-compose up -d
```

### Notification Integration

Uncomment and configure notification commands:

```yaml
# Slack notification
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚀 Deployment successful!"}' \
  ${{ secrets.SLACK_WEBHOOK }}
```

### Performance Testing

Customize performance benchmarks in the CI workflow:

```yaml
python scripts/performance_benchmark.py --output results.json
```

## Monitoring

The workflows include comprehensive monitoring and reporting:

- **Test results** and coverage reports
- **Security scan** results
- **Performance benchmark** results
- **Dependency analysis** reports
- **Build artifacts** for releases

Access these through GitHub Actions artifacts and the integrated reporting tools.

## Troubleshooting

### Common Issues

1. **Workflow Permission Errors**:
   - Ensure your repository has the necessary permissions
   - Check if the GitHub App has `workflows` permission

2. **Docker Build Failures**:
   - Verify Dockerfile exists and is valid
   - Check if all dependencies are available

3. **Test Failures**:
   - Review test configuration in pyproject.toml
   - Ensure test dependencies are installed

4. **Deployment Failures**:
   - Verify environment secrets are configured
   - Check deployment commands match your infrastructure

### Getting Help

- Review GitHub Actions logs for detailed error messages
- Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
- Review the project's development documentation

---

**Note**: These workflows are designed to work with the comprehensive SDLC setup implemented in the repository. Make sure to run `make setup` to configure your development environment properly.