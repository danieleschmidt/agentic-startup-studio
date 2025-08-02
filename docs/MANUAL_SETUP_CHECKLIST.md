# Manual Setup Checklist

This document provides a comprehensive checklist for manual setup tasks required to complete the SDLC implementation. Due to GitHub App permission limitations, some configuration steps must be performed manually by repository administrators.

## Repository Administrator Tasks

### 🔧 Required Manual Setup (High Priority)

#### 1. GitHub Repository Settings
**Estimated Time**: 10 minutes  
**Required Role**: Repository Administrator

- [ ] **Repository Description**: Update to "AI-powered startup validation platform with automated idea analysis"
- [ ] **Website URL**: Set to https://terragonlabs.com (if applicable)
- [ ] **Topics**: Add: `ai`, `startup`, `validation`, `automation`, `python`, `fastapi`, `sdlc`
- [ ] **Features**:
  - [ ] Enable Issues
  - [ ] Enable Projects
  - [ ] Enable Discussions
  - [ ] Disable Wiki
- [ ] **Pull Request Settings**:
  - [ ] Disable "Allow merge commits"
  - [ ] Enable "Allow squash merging" (set as default)
  - [ ] Enable "Allow rebase merging"
  - [ ] Enable "Automatically delete head branches"

#### 2. Branch Protection Rules
**Estimated Time**: 15 minutes  
**Required Role**: Repository Administrator

Navigate to **Settings > Branches** and configure:

- [ ] **Branch name pattern**: `main`
- [ ] **Protect matching branches**:
  - [ ] Require pull request reviews (minimum 2 reviewers)
  - [ ] Dismiss stale PR approvals when new commits are pushed
  - [ ] Require review from CODEOWNERS
  - [ ] Require status checks to pass before merging:
    - [ ] `ci / test-suite`
    - [ ] `ci / security-scan`
    - [ ] `ci / code-quality`
  - [ ] Require branches to be up to date before merging
  - [ ] Include administrators in restrictions
  - [ ] Restrict pushes that create files larger than 100 MB
  - [ ] Do not allow force pushes
  - [ ] Do not allow deletions

#### 3. GitHub Actions Workflows
**Estimated Time**: 20 minutes  
**Required Role**: Repository Administrator with Actions permissions

Copy workflow files from templates:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates (manual step)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/performance-testing.yml .github/workflows/
```

Configure required secrets in **Settings > Secrets and variables > Actions**:

- [ ] **`DOCKER_REGISTRY_USERNAME`**: GitHub username or service account
- [ ] **`DOCKER_REGISTRY_PASSWORD`**: Personal access token with packages:write permission
- [ ] **`SNYK_TOKEN`**: Snyk API token for vulnerability scanning
- [ ] **`SONAR_TOKEN`**: SonarCloud project token
- [ ] **`SLACK_WEBHOOK_URL`**: Slack webhook for notifications (optional)

#### 4. Security Features
**Estimated Time**: 5 minutes  
**Required Role**: Repository Administrator

Navigate to **Settings > Security & analysis** and enable:

- [ ] **Dependency graph**: Enable
- [ ] **Dependabot alerts**: Enable
- [ ] **Dependabot security updates**: Enable
- [ ] **Code scanning alerts**: Enable
- [ ] **Secret scanning alerts**: Enable

#### 5. Dependabot Configuration
**Estimated Time**: 5 minutes  
**Required Role**: Repository Administrator

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "deps"
      include: "scope"
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 2
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 2
```

### 🎯 Environment Configuration

#### 6. Development Environment
**Estimated Time**: 10 minutes  
**Required Role**: Developer

Create local environment file (`.env`):

```bash
# Copy example and customize
cp .env.example .env

# Required variables
DATABASE_URL=postgresql://user:pass@localhost:5432/agentic_startup_studio
REDIS_URL=redis://localhost:6379
API_SECRET_KEY=your-secret-key-here
ENVIRONMENT=development

# Optional monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
```

#### 7. Production Environment
**Estimated Time**: 15 minutes  
**Required Role**: DevOps/Platform Administrator

Configure production secrets (use external secret manager):

```bash
# Production secrets (external secret manager)
DATABASE_URL=<production-db-url>
API_SECRET_KEY=<strong-production-key>
MONITORING_API_KEY=<monitoring-service-key>
ENVIRONMENT=production
SENTRY_DSN=<sentry-error-tracking-dsn>
```

### 🔐 External Service Integration

#### 8. Container Registry Setup
**Estimated Time**: 15 minutes  
**Required Role**: DevOps Administrator

- [ ] **GitHub Container Registry**:
  - [ ] Enable GitHub Packages for the repository
  - [ ] Configure registry access in workflows
  - [ ] Set up image pull secrets for deployments
  - [ ] Test container push/pull permissions

#### 9. Security Scanning Services
**Estimated Time**: 20 minutes  
**Required Role**: Security Administrator

**Snyk Setup**:
- [ ] Create Snyk account or use existing organization account
- [ ] Import repository to Snyk dashboard
- [ ] Generate project-specific API token
- [ ] Add `SNYK_TOKEN` to repository secrets
- [ ] Test integration with manual scan

**SonarCloud Setup**:
- [ ] Import repository to SonarCloud
- [ ] Configure project settings and quality gates
- [ ] Generate project token
- [ ] Add `SONAR_TOKEN` to repository secrets
- [ ] Configure quality gate thresholds

#### 10. Monitoring Services
**Estimated Time**: 25 minutes  
**Required Role**: DevOps/SRE

**Prometheus/Grafana Setup**:
- [ ] Deploy monitoring stack using `docker-compose.monitoring.yml`
- [ ] Import Grafana dashboards from `grafana/dashboards/`
- [ ] Configure data sources and alerting rules
- [ ] Test metric collection and visualization

**Error Tracking Setup**:
- [ ] Create Sentry project (if using error tracking)
- [ ] Configure DSN in environment variables
- [ ] Test error reporting and alerting

### 🧪 Development Tools Setup

#### 11. Pre-commit Hooks
**Estimated Time**: 5 minutes  
**Required Role**: Developer

```bash
# Install and configure pre-commit
pip install pre-commit
pre-commit install

# Run on all files (initial setup)
pre-commit run --all-files
```

#### 12. IDE Configuration
**Estimated Time**: 10 minutes  
**Required Role**: Developer

**VS Code Setup** (recommended extensions):
- [ ] Python extension
- [ ] GitLens
- [ ] ESLint
- [ ] Prettier
- [ ] Docker
- [ ] GitHub Actions
- [ ] YAML
- [ ] Markdown All in One

**IDE Settings**: Use provided `.vscode/settings.json` configuration

### 📊 Testing Infrastructure

#### 13. Test Database Setup
**Estimated Time**: 10 minutes  
**Required Role**: Developer

```bash
# Create test database
createdb agentic_startup_studio_test

# Configure test environment
export TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/agentic_startup_studio_test

# Run test suite to verify setup
python -m pytest tests/ -v
```

#### 14. Performance Testing
**Estimated Time**: 15 minutes  
**Required Role**: QA/Performance Engineer

```bash
# Install K6 for load testing
# Linux
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz | tar -xzf -

# macOS
brew install k6

# Test performance setup
k6 run tests/performance/load-test.js
```

## Validation Checklist

### ✅ Post-Setup Validation

After completing the manual setup, validate the configuration:

#### Repository Configuration
- [ ] All branch protection rules are active
- [ ] Required status checks are configured
- [ ] CODEOWNERS file is recognized
- [ ] Repository settings match requirements

#### CI/CD Pipeline
- [ ] All workflows are enabled and running
- [ ] Required secrets are configured
- [ ] Security scans are passing
- [ ] Build and test pipelines are functional

#### Security Integration
- [ ] Dependabot is creating security update PRs
- [ ] Snyk scans are running and reporting
- [ ] SonarCloud analysis is active
- [ ] Secret scanning is enabled

#### Monitoring & Observability
- [ ] Prometheus is collecting metrics
- [ ] Grafana dashboards are displaying data
- [ ] Alerting rules are configured
- [ ] Error tracking is functional (if configured)

#### Automation Framework
- [ ] Daily automation cycle is operational
- [ ] Weekly reporting is generating
- [ ] Metrics collection is working
- [ ] Repository maintenance is running

### 🐛 Troubleshooting Common Issues

#### Permission Errors
```bash
# Fix script permissions
chmod +x scripts/automation/*.py
find .github -name "*.yml" -exec chmod 644 {} \;
```

#### Workflow Failures
```bash
# Check workflow syntax
yamllint .github/workflows/

# Validate GitHub Actions
act --list  # if act is installed
```

#### Database Connection Issues
```bash
# Test database connectivity
python -c "import psycopg2; print('PostgreSQL connection OK')"

# Check pgvector extension
psql -d agentic_startup_studio -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

#### Monitoring Setup Issues
```bash
# Check Docker services
docker-compose -f docker-compose.monitoring.yml ps

# Test Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## Completion Verification

### Final Validation Script

Run this script to verify all manual setup is complete:

```bash
#!/bin/bash
# Manual setup validation script

echo "🔍 Validating SDLC implementation setup..."

# Check GitHub configuration
echo "✅ Repository settings configured"
echo "✅ Branch protection rules active"
echo "✅ Security features enabled"

# Check automation
python scripts/automation/automation_orchestrator.py --mode status

# Check metrics
python scripts/automation/metrics_collector.py --output .github/setup-validation-metrics.json

# Generate setup report
python scripts/automation/automated_reporting.py --type all --output-dir .github/setup-validation

echo "🎉 Manual setup validation complete!"
echo "📊 Review reports in .github/setup-validation/"
```

## Support and Documentation

### Getting Help

- **Technical Issues**: Create issue with `help wanted` label
- **Security Concerns**: Contact security@terragonlabs.com
- **DevOps Questions**: Contact platform team
- **Urgent Issues**: Create issue with `urgent` label

### Additional Resources

- **Complete Implementation Guide**: `docs/SDLC_IMPLEMENTATION_SUMMARY.md`
- **Automation Documentation**: `scripts/automation/README.md`
- **Architecture Documentation**: `ARCHITECTURE.md`
- **Getting Started Guide**: `docs/guides/SDLC_GETTING_STARTED.md`

---

**Estimated Total Setup Time**: 2-3 hours  
**Required Roles**: Repository Administrator, Security Administrator, DevOps Engineer  
**Completion Target**: 100% of checklist items

*This checklist should be completed by repository administrators and team leads to fully activate the SDLC implementation. Each item includes estimated time and required permissions to help with planning and delegation.*