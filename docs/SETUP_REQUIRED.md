# Enterprise SDLC Manual Setup Requirements

This document outlines the comprehensive manual configuration steps required to complete the enterprise-grade SDLC implementation. These steps require elevated permissions or external configurations that cannot be automated due to GitHub App limitations.

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

### 2. Copy Enterprise Workflow Files
```bash
# Copy enterprise workflows from templates (manual step required)
cp docs/workflows/examples/enterprise-ci.yml .github/workflows/
cp docs/workflows/examples/enterprise-cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/performance-testing.yml .github/workflows/
```

### 3. Configure Repository Secrets
Add secrets in **Settings > Secrets and variables > Actions**:

#### Enterprise Required Secrets
```bash
# Container Registry & Deployment
REGISTRY_USERNAME=your-github-username
REGISTRY_PASSWORD=your-github-pat-with-packages-write
STAGING_DEPLOY_KEY=staging-ssh-private-key
PRODUCTION_DEPLOY_KEY=production-ssh-private-key
KUBECONFIG_STAGING=base64-encoded-staging-kubeconfig
KUBECONFIG_PRODUCTION=base64-encoded-production-kubeconfig

# Security & Compliance
SNYK_TOKEN=your-snyk-api-token
SONARCLOUD_TOKEN=your-sonarcloud-project-token
SECURITY_EMAIL=security@terragonlabs.com

# Monitoring & Observability
PROMETHEUS_ENDPOINT=https://prometheus.monitoring.terragonlabs.com
GRAFANA_API_KEY=your-grafana-api-key
DATADOG_API_KEY=your-datadog-api-key
NEW_RELIC_LICENSE_KEY=your-newrelic-license-key

# Notifications
SLACK_WEBHOOK_URL=your-slack-webhook-url
DISCORD_WEBHOOK_URL=your-discord-webhook-url
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_USERNAME=alerts@terragonlabs.com
EMAIL_SMTP_PASSWORD=your-app-password
```

#### Environment-Specific Secrets
```bash
# Staging Environment Variables
STAGING_DATABASE_URL=postgresql://staging-db-connection
STAGING_REDIS_URL=redis://staging-cache-connection
STAGING_API_BASE_URL=https://staging.agentic-startup-studio.terragonlabs.com

# Production Environment Variables  
PRODUCTION_DATABASE_URL=postgresql://production-db-connection
PRODUCTION_REDIS_URL=redis://production-cache-connection
PRODUCTION_API_BASE_URL=https://agentic-startup-studio.terragonlabs.com
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

## Enterprise Infrastructure Setup

### 1. Kubernetes Cluster Configuration

```yaml
# Required Kubernetes resources (apply manually)

# Namespaces
apiVersion: v1
kind: Namespace
metadata:
  name: staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: production
---
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring

# Service Accounts for CI/CD
apiVersion: v1
kind: ServiceAccount
metadata:
  name: github-actions-deployer
  namespace: staging
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: github-actions-deployer
  namespace: production
```

### 2. Environment Protection Rules

Configure in **Settings > Environments**:

```yaml
# Staging Environment
staging:
  protection_rules:
    required_reviewers: []
    wait_timer: 0
    prevent_self_review: false
  deployment_branches:
    - main
    - develop

# Production Environment  
production:
  protection_rules:
    required_reviewers: ["@team-leads", "@security-team"]
    wait_timer: 300  # 5 minutes
    prevent_self_review: true
  deployment_branches:
    - main
    - release/*
```

### 3. Monitoring Stack Deployment

```bash
# Deploy Prometheus and Grafana to Kubernetes
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml

# Install Grafana (if not included in kube-prometheus-stack)
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values monitoring/grafana-values.yaml
```

### 4. SSL/TLS Certificate Setup

```bash
# Install cert-manager for automatic SSL certificates
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Configure Let's Encrypt ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: devops@terragonlabs.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Enterprise Security Configuration

### 1. Advanced Security Scanning

```bash
# Snyk Integration Setup
1. Sign up at https://snyk.io
2. Import GitHub repository
3. Generate API token from account settings
4. Add SNYK_TOKEN to GitHub repository secrets

# SonarCloud Integration Setup  
1. Sign up at https://sonarcloud.io
2. Import repository from GitHub
3. Configure quality gates and rules
4. Generate project token
5. Add SONARCLOUD_TOKEN to GitHub repository secrets
```

### 2. Compliance Monitoring Setup

```bash
# SOC 2 Compliance Setup
1. Configure audit logging for all system access
2. Set up access controls and role-based permissions
3. Enable encryption at rest and in transit
4. Configure backup and disaster recovery procedures

# GDPR Compliance Setup
1. Implement data minimization practices
2. Set up data deletion procedures
3. Configure privacy controls
4. Enable audit trails for data access

# ISO 27001 Alignment
1. Document information security policies
2. Implement risk management procedures
3. Set up incident response procedures
4. Configure regular security assessments
```

### 3. Container Security

```bash
# Container scanning with Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image your-image:tag

# SBOM (Software Bill of Materials) generation
# This is automated via scripts/build_sbom.py
python scripts/build_sbom.py --format spdx --output sbom.json
```

## Database and Data Management

### 1. Database Setup

```sql
-- Create application databases
CREATE DATABASE agentic_startup_studio_dev;
CREATE DATABASE agentic_startup_studio_staging;  
CREATE DATABASE agentic_startup_studio_production;

-- Create dedicated user accounts
CREATE USER app_staging WITH PASSWORD 'secure_staging_password';
CREATE USER app_production WITH PASSWORD 'secure_production_password';
CREATE USER monitoring_user WITH PASSWORD 'monitoring_password';

-- Grant appropriate permissions
GRANT ALL PRIVILEGES ON DATABASE agentic_startup_studio_staging TO app_staging;
GRANT ALL PRIVILEGES ON DATABASE agentic_startup_studio_production TO app_production;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring_user;
```

### 2. Backup Configuration

```bash
# Set up automated database backups
# 1. Configure pg_dump scripts with encryption
# 2. Set up S3 or GCS backup storage
# 3. Configure backup retention policies
# 4. Test restoration procedures

# Example backup script (customize for your environment)
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | \
  gzip | \
  gpg --cipher-algo AES256 --compress-algo 1 --symmetric --output backup_$(date +%Y%m%d_%H%M%S).sql.gz.gpg
```

### 3. Cache Configuration

```bash
# Redis setup for caching
# 1. Deploy Redis cluster for high availability
# 2. Configure authentication and encryption
# 3. Set up monitoring and alerting
# 4. Configure backup procedures
```

## Load Balancer and CDN Setup

### 1. Load Balancer Configuration

```nginx
# Example Nginx configuration for load balancing
upstream backend {
    least_conn;
    server staging.agentic-startup-studio.terragonlabs.com:8000;
    server staging-backup.agentic-startup-studio.terragonlabs.com:8000 backup;
}

server {
    listen 443 ssl http2;
    server_name staging.agentic-startup-studio.terragonlabs.com;

    ssl_certificate /etc/ssl/certs/staging.crt;
    ssl_certificate_key /etc/ssl/private/staging.key;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://backend/health;
    }
}
```

### 2. CDN Configuration

```bash
# CloudFlare or AWS CloudFront setup
# 1. Configure origin servers
# 2. Set up caching rules for static assets
# 3. Configure security headers
# 4. Set up cache invalidation triggers
# 5. Configure DDoS protection
```

## Notification and Communication Setup

### 1. Slack Integration

```bash
# Slack Webhook Setup
1. Go to your Slack workspace settings
2. Navigate to Apps & Integrations
3. Add "Incoming Webhooks" app
4. Create webhook for each channel:
   - #deployments
   - #security
   - #monitoring  
   - #production
5. Add webhook URLs to GitHub secrets
```

### 2. Email Notification Setup

```bash
# SMTP Configuration for alerts
# 1. Set up dedicated email account for system alerts
# 2. Generate app-specific passwords
# 3. Configure email templates for different alert types
# 4. Set up email routing rules
# 5. Configure spam filters and authentication
```

### 3. PagerDuty Integration (Optional)

```bash
# For production incident management
# 1. Create PagerDuty account
# 2. Set up escalation policies
# 3. Configure integration key
# 4. Set up on-call rotations
# 5. Configure alert routing rules
```

## Enterprise Compliance and Auditing

### 1. Audit Log Configuration

```bash
# GitHub Enterprise Audit Log API setup
# 1. Enable audit log streaming
# 2. Configure log forwarding to SIEM
# 3. Set up log retention policies
# 4. Configure audit alert rules
# 5. Set up compliance reporting
```

### 2. Access Control and Identity Management

```bash
# SAML/SSO Integration
# 1. Configure identity provider (Okta, Azure AD, etc.)
# 2. Set up SAML assertions
# 3. Configure role-based access controls
# 4. Set up multi-factor authentication
# 5. Configure session management
```

### 3. Data Governance

```bash
# Data classification and handling
# 1. Classify data sensitivity levels
# 2. Implement data retention policies
# 3. Set up data lineage tracking
# 4. Configure data anonymization procedures
# 5. Set up data breach response procedures
```

## Disaster Recovery and Business Continuity

### 1. Backup and Recovery

```bash
# Multi-region backup strategy
# 1. Configure database replication across regions
# 2. Set up application state backups
# 3. Configure infrastructure as code backups
# 4. Set up recovery time objectives (RTO)
# 5. Configure recovery point objectives (RPO)
```

### 2. Failover Procedures

```bash
# Automated failover setup
# 1. Configure health checks and monitoring
# 2. Set up automatic DNS failover
# 3. Configure database failover procedures
# 4. Set up application failover automation
# 5. Document manual failover procedures
```

## Performance and Scalability

### 1. Auto-scaling Configuration

```bash
# Kubernetes Horizontal Pod Autoscaler
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-startup-studio
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

### 2. Performance Monitoring

```bash
# Application Performance Monitoring setup
# 1. Configure APM tools (New Relic, DataDog, etc.)
# 2. Set up custom metrics and dashboards
# 3. Configure performance alerts
# 4. Set up synthetic monitoring
# 5. Configure user experience monitoring
```

## Enterprise Checklist

### Core Infrastructure ✅
- [ ] Kubernetes cluster configured with proper namespaces
- [ ] Environment protection rules configured in GitHub
- [ ] SSL/TLS certificates installed and auto-renewing
- [ ] Load balancer configured with health checks
- [ ] CDN configured for static assets

### Security & Compliance ✅
- [ ] All required security scanning tools integrated
- [ ] Compliance monitoring active (SOC 2, GDPR, ISO 27001)
- [ ] Container security scanning implemented
- [ ] SBOM generation automated
- [ ] Audit logging configured and tested

### Monitoring & Observability ✅
- [ ] Prometheus metrics collection active
- [ ] Grafana dashboards configured
- [ ] Alert manager rules configured
- [ ] Log aggregation working
- [ ] APM tools integrated

### Data Management ✅
- [ ] Production databases configured with encryption
- [ ] Backup procedures tested and automated
- [ ] Cache infrastructure deployed
- [ ] Data retention policies implemented
- [ ] Disaster recovery procedures tested

### CI/CD & Automation ✅
- [ ] All enterprise workflows copied and functional
- [ ] Deployment automation tested
- [ ] Blue-green deployment procedures working
- [ ] Rollback procedures tested
- [ ] Automated testing pipeline complete

### Communication & Notifications ✅
- [ ] Slack integrations configured for all channels
- [ ] Email notifications working
- [ ] PagerDuty integration active (if applicable)
- [ ] Escalation procedures documented
- [ ] Emergency contact procedures tested

## Enterprise Support Contacts

### Immediate Response (24/7)
- **Production Incidents**: incidents@terragonlabs.com
- **Security Incidents**: security@terragonlabs.com  
- **Emergency Hotline**: +1-XXX-XXX-XXXX

### Standard Support (Business Hours)
- **Infrastructure Issues**: devops@terragonlabs.com
- **Development Support**: dev-support@terragonlabs.com
- **Compliance Questions**: compliance@terragonlabs.com
- **General Questions**: support@terragonlabs.com

### Escalation Paths
1. **Level 1**: Development team lead
2. **Level 2**: Platform engineering manager
3. **Level 3**: VP of Engineering
4. **Level 4**: CTO

---

**Document Version**: 2.0 (Enterprise Edition)  
**Last Updated**: 2025-08-02  
**Next Review**: 2025-09-02  
**Maintained By**: Terragon Labs Platform Engineering Team  

*This document should be updated whenever new enterprise setup requirements are identified. All manual setup steps must be validated and documented with screenshots or configuration files.*