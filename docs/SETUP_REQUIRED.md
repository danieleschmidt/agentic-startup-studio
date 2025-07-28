# Manual Setup Requirements

This document outlines the manual setup steps required to complete the SDLC implementation for this repository. These steps require elevated permissions or external configurations that cannot be automated.

## GitHub Repository Configuration

### 1. Branch Protection Rules
Configure branch protection in **Settings > Branches**:

```bash
# Main branch protection settings
- Branch name pattern: main
- Require pull request reviews: ✅ (2 reviewers minimum)
- Dismiss stale PR approvals: ✅
- Require review from CODEOWNERS: ✅
- Require status checks: ✅
  - ci / test-suite
  - ci / security-scan  
  - ci / code-quality
- Require branches to be up to date: ✅
- Include administrators: ✅
- Allow force pushes: ❌
- Allow deletions: ❌
```

### 2. Repository Settings
Update repository settings in **Settings > General**:

```bash
# Repository details
Description: "AI-powered startup validation platform with automated idea analysis"
Website: https://terragonlabs.com
Topics: ai, startup, validation, automation, python, fastapi

# Features
- Issues: ✅
- Projects: ✅  
- Wiki: ❌
- Discussions: ✅

# Pull Requests
- Allow merge commits: ❌
- Allow squash merging: ✅ (default)
- Allow rebase merging: ✅
- Auto-delete head branches: ✅
```

## GitHub Actions Setup

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Files
```bash
# Copy from templates (manual step required)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
```

### 3. Configure Repository Secrets
Add secrets in **Settings > Secrets and variables > Actions**:

#### Required Secrets
```bash
# Docker Registry
DOCKER_REGISTRY_URL=ghcr.io
DOCKER_REGISTRY_USERNAME=${{ github.actor }}
DOCKER_REGISTRY_PASSWORD=${{ secrets.GITHUB_TOKEN }}

# Security Scanning
SNYK_TOKEN=your-snyk-token
SONAR_TOKEN=your-sonar-token

# Notifications
SLACK_WEBHOOK_URL=your-slack-webhook-url
```

#### Optional Secrets (for enhanced features)
```bash
# Cloud Deployment
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## Security Configuration

### 1. Enable Security Features
Configure in **Settings > Security & analysis**:

```bash
# Dependency graph: ✅
# Dependabot alerts: ✅
# Dependabot security updates: ✅
# Code scanning alerts: ✅
# Secret scanning alerts: ✅
```

### 2. Dependabot Configuration
Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

### 3. CODEOWNERS File
Create `.github/CODEOWNERS`:
```bash
# Global owners
* @your-username @security-team

# Security-sensitive files
/SECURITY.md @security-team
/.github/ @security-team @devops-team
/scripts/security_scan.py @security-team
```

## Environment Configuration

### 1. Development Environment
```bash
# Required environment variables (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379
API_SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
```

### 2. Production Environment  
```bash
# Production secrets (external secret manager)
DATABASE_URL=<production-db-url>
API_SECRET_KEY=<strong-production-key>
MONITORING_API_KEY=<monitoring-service-key>
ENVIRONMENT=production
```

## External Service Integration

### 1. Container Registry Setup
```bash
# GitHub Container Registry (recommended)
1. Enable GitHub Packages
2. Configure registry access in workflows
3. Set up image pull secrets for deployments
```

### 2. Monitoring Services
```bash
# Sentry for error tracking
SENTRY_DSN=your-sentry-dsn

# Prometheus/Grafana for metrics
PROMETHEUS_URL=your-prometheus-url
GRAFANA_API_KEY=your-grafana-key
```

### 3. Security Scanning Services
```bash
# Snyk for vulnerability scanning
1. Create Snyk account
2. Generate API token
3. Add SNYK_TOKEN to repository secrets

# SonarCloud for code quality
1. Import repository to SonarCloud  
2. Generate project token
3. Add SONAR_TOKEN to repository secrets
```

## Development Tools Setup

### 1. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (initial setup)
pre-commit run --all-files
```

### 2. IDE Configuration
```bash
# VS Code extensions (recommended)
- Python
- GitLens
- ESLint
- Prettier
- Docker
- GitHub Actions
```

## Testing Infrastructure

### 1. Test Database Setup
```bash
# Create test database
createdb agentic_startup_studio_test

# Configure test environment
export TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test_db
```

### 2. Performance Testing
```bash
# Install K6 for load testing
curl https://github.com/grafana/k6/releases/.../k6-linux-amd64.tar.gz | tar -xzf -
```

## Deployment Configuration

### 1. Container Orchestration
```bash
# Kubernetes manifests (if using K8s)
kubectl apply -f k8s/

# Docker Compose for local development
docker-compose -f docker-compose.dev.yml up
```

### 2. Domain and SSL
```bash
# Domain configuration
1. Configure DNS records
2. Set up SSL certificates
3. Configure load balancer
```

## Compliance and Auditing

### 1. Audit Logging
```bash
# Configure audit log collection
1. Enable GitHub audit log API
2. Set up log forwarding to SIEM
3. Configure retention policies
```

### 2. Backup Configuration  
```bash
# Database backups
1. Configure automated backups
2. Set up cross-region replication
3. Test backup restoration procedures
```

## Checklist

### Repository Setup ✅
- [ ] Branch protection rules configured
- [ ] Repository settings updated
- [ ] Topics and description added
- [ ] Security features enabled

### GitHub Actions ✅  
- [ ] Workflow files copied
- [ ] Repository secrets configured
- [ ] Dependabot enabled
- [ ] CODEOWNERS file created

### Development Environment ✅
- [ ] Pre-commit hooks installed
- [ ] IDE configured
- [ ] Environment variables set
- [ ] Test database created

### External Services ✅
- [ ] Container registry configured
- [ ] Monitoring services integrated
- [ ] Security scanning enabled
- [ ] Notification channels set up

### Security & Compliance ✅
- [ ] Audit logging configured
- [ ] Backup procedures tested
- [ ] Access controls verified
- [ ] Security scanning active

## Support

For questions about manual setup:
- **Development**: Contact development team lead
- **Security**: Contact security@terragonlabs.com  
- **Infrastructure**: Contact platform team
- **Urgent Issues**: Create issue with `urgent` label

---

*This document should be updated whenever new manual setup requirements are identified.*