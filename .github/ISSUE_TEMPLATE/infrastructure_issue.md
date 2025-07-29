---
name: 🛠️ Infrastructure Issue
about: Report infrastructure, deployment, or operational issues
title: '[INFRA] '
labels: ['infrastructure', 'ops', 'needs-investigation']
assignees: ['terragonlabs/infrastructure-team']
---

## Infrastructure Issue Description

**Affected Infrastructure:**
- [ ] CI/CD Pipeline
- [ ] Container/Docker
- [ ] Database Infrastructure
- [ ] Monitoring/Observability
- [ ] Networking
- [ ] Security Infrastructure
- [ ] Cloud Resources
- [ ] Other: ___________

**Issue Summary:**
A clear and concise description of the infrastructure issue.

## System Impact

**Service Availability:**
- [ ] Complete outage
- [ ] Partial service degradation
- [ ] Intermittent issues
- [ ] No user impact (internal only)

**Affected Environments:**
- [ ] Production
- [ ] Staging
- [ ] Development
- [ ] CI/CD
- [ ] Local development

## Technical Details

**Error Messages:**
```
[Paste error messages, logs, or stack traces here]
```

**System Metrics:**
- CPU Usage: _____ %
- Memory Usage: _____ %
- Disk Usage: _____ %
- Network I/O: _____
- Database Connections: _____

**Timeline:**
- Issue first observed: _____ (timestamp)
- Duration: _____
- Frequency: _____ (if recurring)

## Environment Information

**Infrastructure Stack:**
- Platform: _____ (AWS/GCP/Azure/Local)
- Container Runtime: _____ (Docker/Podman)
- Orchestration: _____ (Docker Compose/Kubernetes)
- Database: PostgreSQL version _____
- Monitoring: _____ (Prometheus/Grafana)
- CI/CD: _____ (GitHub Actions/Other)

**Configuration Details:**
- Resource limits: _____
- Network configuration: _____
- Security settings: _____
- Environment variables: _____ (sensitive data)

## Reproduction Steps

1. Setup environment: ...
2. Deploy/configure: ...
3. Trigger condition: ...
4. Observe failure: ...

## Investigation Data

**Logs Available:** (attach relevant logs)
- [ ] Application logs
- [ ] System logs
- [ ] Container logs
- [ ] Database logs
- [ ] CI/CD logs
- [ ] Monitoring alerts

**Monitoring Data:**
- [ ] Grafana dashboard screenshots
- [ ] Prometheus metrics
- [ ] Health check results
- [ ] Performance metrics

## Immediate Actions Taken

**Mitigation Steps:**
1. _____
2. _____
3. _____

**Current Status:**
- [ ] Issue resolved temporarily
- [ ] Workaround in place
- [ ] Issue ongoing
- [ ] Root cause identified

## Proposed Solutions

**Short-term Fixes:**
1. _____
2. _____

**Long-term Improvements:**
1. _____
2. _____

**Infrastructure Changes Needed:**
- Configuration updates: _____
- Resource scaling: _____
- Architecture changes: _____
- Monitoring improvements: _____

## Business Impact

**Impact Level:**
- [ ] Critical - Complete service disruption
- [ ] High - Major functionality affected
- [ ] Medium - Some features impacted
- [ ] Low - Minor operational issue

**Affected Operations:**
- User-facing services: _____
- Internal operations: _____
- Development workflow: _____
- Data integrity: _____

## Additional Context

**Related Issues:**
- #_____ (infrastructure issues)
- #_____ (related incidents)

**Recent Changes:**
- [ ] Recent deployments
- [ ] Configuration changes
- [ ] Infrastructure updates
- [ ] Third-party service changes

**Supporting Documentation:**
- [ ] Architecture diagrams
- [ ] Runbooks
- [ ] Configuration files
- [ ] Monitoring setup

---

### For Maintainers

**Infrastructure Review Checklist:**
- [ ] Verified issue reproduction
- [ ] Analyzed system metrics
- [ ] Reviewed recent changes
- [ ] Assessed business impact
- [ ] Identified root cause
- [ ] Created incident report (if applicable)
- [ ] Updated monitoring/alerting
- [ ] Documented resolution steps