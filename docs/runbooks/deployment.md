# Deployment Runbook
*Terragon Labs Agentic Startup Studio*

> **🎯 GOAL**: Safe, reliable, and automated deployments with zero-downtime and rollback capabilities

## Deployment Strategy Overview

We implement **Blue-Green deployments** with automated health checks and staged rollouts to minimize risk and ensure high availability.

### Deployment Environments

| Environment | Purpose | Auto-Deploy | Approval Required |
|-------------|---------|-------------|-------------------|
| **Development** | Feature development and testing | ✅ On every commit to `dev` | None |
| **Staging** | Integration testing and QA | ✅ On PR to `main` | QA Team |
| **Production** | Live customer-facing service | ❌ Manual trigger only | SRE + Engineering Manager |

---

## Pre-Deployment Checklist

### Automated Checks (CI Pipeline)
- [ ] All tests pass (unit, integration, e2e)
- [ ] Security scans complete (SAST, dependency check)
- [ ] Code coverage meets 90% threshold
- [ ] Performance benchmarks within acceptable range
- [ ] Database migrations tested and reviewed
- [ ] API compatibility verified (no breaking changes)

### Manual Verification
- [ ] Feature flags configured properly
- [ ] Configuration secrets updated in vault
- [ ] Monitoring dashboards prepared for new metrics
- [ ] Runbook updated with any new procedures
- [ ] Rollback plan confirmed and tested
- [ ] Customer communication prepared (if user-facing changes)

---

## Deployment Procedures

### 1. Development Deployment (Automatic)

**Trigger**: Every commit to `dev` branch

```bash
# Automatic via GitHub Actions
# .github/workflows/deploy-dev.yml

# Manual deployment (if needed)
git checkout dev
git pull origin dev
./scripts/deploy.sh --environment=dev --auto-approve
```

**Validation**:
```bash
# Health check
curl -f https://dev-api.terragonlabs.com/health

# Smoke test
python scripts/run_smoke_test.py --environment=dev
```

### 2. Staging Deployment

**Trigger**: PR created/updated against `main` branch

```bash
# Automatic staging deployment via PR
git checkout main
git pull origin main
git checkout -b feature/new-deployment
git push origin feature/new-deployment
# Create PR - staging deployment happens automatically

# Manual staging deployment
./scripts/deploy.sh --environment=staging --confirm
```

**Validation**:
```bash
# Comprehensive testing
python scripts/run_comprehensive_tests.py --environment=staging

# Load testing
python tests/performance/load-test.js --environment=staging

# Manual QA testing
echo "Run through QA checklist in staging environment"
```

### 3. Production Deployment

**Trigger**: Manual execution after staging approval

#### Step 3.1: Pre-Production Setup
```bash
# Set production environment
export ENVIRONMENT=production
export DEPLOYMENT_ID=$(date +%Y%m%d-%H%M%S)

# Verify staging is healthy
./scripts/validate_deployment.py --environment=staging --strict

# Check production readiness
./scripts/run_health_checks.py --environment=production --pre-deployment

# Verify secrets and configuration
./scripts/validate_production_secrets.py --verify-all
```

#### Step 3.2: Blue-Green Deployment
```bash
# Start deployment (creates green environment)
./scripts/deploy.sh --environment=production --strategy=blue-green --deployment-id=$DEPLOYMENT_ID

# This will:
# 1. Create new "green" environment with updated code
# 2. Run health checks on green environment  
# 3. Gradually shift traffic from blue to green
# 4. Monitor key metrics during transition
# 5. Complete switch or auto-rollback on issues
```

#### Step 3.3: Post-Deployment Validation
```bash
# Wait for deployment completion
echo "Deployment initiated. Monitor progress:"
echo "Dashboard: https://grafana.terragonlabs.com/d/deployment"
echo "Logs: kubectl logs -f deployment/api-gateway -n production"

# Automated post-deployment checks (runs automatically)
./scripts/validate_deployment.py --environment=production --post-deployment

# Health check endpoints
curl -f https://api.terragonlabs.com/health
curl -f https://api.terragonlabs.com/metrics

# Business logic validation
python scripts/run_smoke_test.py --environment=production --full-suite
```

---

## Blue-Green Deployment Details

### Environment Setup
```bash
# Production environments
BLUE_NAMESPACE="production-blue"
GREEN_NAMESPACE="production-green"
ACTIVE_NAMESPACE=$(kubectl get service main-service -o jsonpath='{.spec.selector.version}')

echo "Current active environment: $ACTIVE_NAMESPACE"
```

### Traffic Switching Process
```bash
# Gradual traffic shift (automated by deployment script)
# Phase 1: 5% traffic to green (canary)
kubectl patch service main-service -p '{"spec":{"selector":{"version":"green"}}}'
kubectl annotate service main-service traffic.weight="green=5,blue=95"

# Monitor for 10 minutes, then Phase 2: 25% traffic
kubectl annotate service main-service traffic.weight="green=25,blue=75"

# Monitor for 15 minutes, then Phase 3: 100% traffic
kubectl annotate service main-service traffic.weight="green=100,blue=0"

# Clean up old blue environment after 24 hours
kubectl delete namespace $BLUE_NAMESPACE
```

### Monitoring During Deployment
```bash
# Key metrics to watch (automated monitoring)
# Error rate should stay < 1%
# Response time should stay < 200ms P95
# CPU/Memory usage should stay < 80%

# Manual monitoring commands
watch -n 5 'kubectl get pods -n production-green | grep -v Running'
watch -n 10 'curl -s https://api.terragonlabs.com/metrics | grep error_rate'
```

---

## Database Migrations

### Migration Safety Checklist
- [ ] Migration is backwards-compatible
- [ ] Migration tested on staging with production-sized data
- [ ] Rollback plan prepared and tested
- [ ] No data loss potential
- [ ] Performance impact assessed
- [ ] Downtime window approved (if required)

### Migration Execution
```bash
# Pre-migration backup
pg_dump $PRODUCTION_DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migration (zero-downtime for compatible changes)
python manage.py migrate --environment=production --dry-run
python manage.py migrate --environment=production

# Verify migration success
python manage.py show_migrations --environment=production
python scripts/validate_database_schema.py --environment=production
```

### Migration Rollback
```bash
# Automatic rollback (if migration fails)
python manage.py migrate --environment=production --rollback

# Manual rollback (if needed later)
python manage.py migrate app_name previous_migration_number --environment=production

# Nuclear option (restore from backup)
# CAUTION: This will cause downtime
pg_restore -d $PRODUCTION_DATABASE_URL backup_YYYYMMDD_HHMMSS.sql
```

---

## Rollback Procedures

### Automatic Rollback Triggers
- Error rate exceeds 2% for more than 5 minutes
- Response time P95 exceeds 500ms for more than 10 minutes
- Health check failures for more than 3 minutes
- CPU/Memory usage exceeds 90% for more than 15 minutes

### Manual Rollback
```bash
# Quick rollback to previous version
./scripts/rollback.sh --environment=production --to-previous

# Rollback to specific version
./scripts/rollback.sh --environment=production --to-version=v2.1.5

# Nuclear rollback (blue-green switch)
kubectl patch service main-service -p '{"spec":{"selector":{"version":"blue"}}}'
kubectl annotate service main-service traffic.weight="blue=100,green=0"
```

### Post-Rollback Actions
```bash
# Verify rollback success
./scripts/validate_deployment.py --environment=production --post-rollback

# Create incident report
echo "Document rollback reason and create incident: INC-$(date +%Y%m%d-%H%M%S)"

# Schedule post-mortem
echo "Schedule post-mortem meeting within 24 hours"
```

---

## Feature Flags and Deployment

### Feature Flag Strategy
```python
# Use feature flags for gradual rollouts
from pipeline.config.feature_flags import FeatureFlags

# In application code
if FeatureFlags.is_enabled("new_ai_model", user_id=user.id):
    return new_ai_model_inference(prompt)
else:
    return legacy_ai_model_inference(prompt)
```

### Feature Flag Management
```bash
# Enable feature for percentage of users
python scripts/feature_flags.py --flag=new_ai_model --enable=10  # 10% of users

# Enable for specific user cohort
python scripts/feature_flags.py --flag=new_ai_model --enable-for-cohort=beta_users

# Full rollout
python scripts/feature_flags.py --flag=new_ai_model --enable=100

# Emergency disable
python scripts/feature_flags.py --flag=new_ai_model --disable
```

---

## Deployment Monitoring

### Key Metrics Dashboard
- **Error Rate**: Should stay < 1% during deployment
- **Response Time**: P95 should stay < 200ms
- **Throughput**: Requests per second should not drop > 10%
- **Resource Usage**: CPU/Memory should stay < 80%
- **Business Metrics**: Pipeline success rate, AI inference accuracy

### Alerting Rules
```yaml
# Example alerting rules (managed in monitoring/alerts.yml)
- alert: DeploymentErrorRateHigh
  expr: increase(http_request_errors_total[5m]) / increase(http_requests_total[5m]) > 0.02
  for: 3m
  labels:
    severity: critical
  annotations:
    summary: "High error rate during deployment"
    description: "Error rate is {{ $value | humanizePercentage }}"

- alert: DeploymentResponseTimeHigh
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High response time during deployment"
```

---

## Emergency Procedures

### Emergency Rollback
If critical production issue is detected:

```bash
# Immediate traffic stop (if necessary)
kubectl scale deployment api-gateway --replicas=0 -n production-green

# Switch all traffic back to blue
kubectl patch service main-service -p '{"spec":{"selector":{"version":"blue"}}}'

# Scale blue back up if needed
kubectl scale deployment api-gateway --replicas=5 -n production-blue

# Verify emergency rollback
curl -f https://api.terragonlabs.com/health
```

### Communication During Emergency
1. **Immediately** notify #engineering-alerts Slack channel
2. **Within 5 minutes** notify #general and customer support
3. **Within 15 minutes** post to status page
4. **Within 30 minutes** send customer email if customer-facing

---

## Deployment Scripts Reference

### Main Deployment Script
```bash
# Usage: ./scripts/deploy.sh [options]
./scripts/deploy.sh --help

# Common deployment patterns
./scripts/deploy.sh --environment=production --strategy=blue-green --confirm
./scripts/deploy.sh --environment=staging --auto-approve
./scripts/deploy.sh --environment=dev --skip-tests  # Emergency only
```

### Validation Scripts
```bash
# Pre-deployment validation
./scripts/validate_deployment.py --environment=production --pre-deployment

# Post-deployment validation  
./scripts/validate_deployment.py --environment=production --post-deployment

# Continuous validation (monitoring)
./scripts/validate_deployment.py --environment=production --continuous --interval=60s
```

### Utility Scripts
```bash
# Health checks
./scripts/run_health_checks.py

# Database migration
./scripts/migrate_database.py --environment=production --preview

# Feature flag management
./scripts/feature_flags.py --list
./scripts/feature_flags.py --flag=new_feature --enable=50

# Performance testing
./scripts/run_performance_tests.py --environment=staging --duration=10m
```

---

## Troubleshooting Common Issues

### Deployment Stuck/Hanging
```bash
# Check deployment status
kubectl rollout status deployment/api-gateway -n production-green --timeout=600s

# Check pod events
kubectl describe pods -n production-green | grep Events -A 10

# Check resource constraints
kubectl top pods -n production-green
kubectl describe node | grep -A 5 "Allocated resources"
```

### Health Checks Failing
```bash
# Check health endpoint directly
kubectl port-forward deployment/api-gateway 8080:8000 -n production-green &
curl -v http://localhost:8080/health

# Check application logs
kubectl logs deployment/api-gateway -n production-green --tail=100

# Check dependencies
kubectl get services -n production-green
nslookup postgres.production.svc.cluster.local
```

### Database Connection Issues
```bash
# Test database connectivity from pod
kubectl exec -it deployment/api-gateway -n production-green -- python -c "
import psycopg2
try:
    conn = psycopg2.connect('$DATABASE_URL')
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database health
psql $DATABASE_URL -c "SELECT 1;"
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"
```

---

*Last updated: $(date) | Review frequency: Monthly | Owner: SRE Team*