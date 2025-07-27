# Operational Runbooks

This directory contains operational runbooks for the Agentic Startup Studio platform. These runbooks provide step-by-step procedures for common operational scenarios, troubleshooting, and incident response.

## Runbook Categories

### 🚨 Incident Response
- [Application Down](./application-down.md) - When the main application is unavailable
- [High Error Rate](./high-error-rate.md) - Dealing with elevated error rates
- [Database Issues](./database-issues.md) - PostgreSQL troubleshooting
- [Performance Degradation](./performance-degradation.md) - Response time issues

### 🔧 Maintenance Procedures
- [Deployment Process](./deployment-process.md) - Safe deployment procedures
- [Backup and Recovery](./backup-recovery.md) - Data backup and restore procedures
- [Security Updates](./security-updates.md) - Applying security patches
- [Scaling Operations](./scaling-operations.md) - Horizontal and vertical scaling

### 📊 Monitoring & Alerting
- [Alert Response](./alert-response.md) - How to respond to specific alerts
- [Monitoring Setup](./monitoring-setup.md) - Setting up new monitoring
- [Log Analysis](./log-analysis.md) - Analyzing application logs
- [Metrics Investigation](./metrics-investigation.md) - Understanding metrics

### 🔒 Security Procedures
- [Security Incident Response](./security-incident-response.md) - Handling security events
- [Access Management](./access-management.md) - User access procedures
- [Vulnerability Management](./vulnerability-management.md) - Handling vulnerabilities
- [Compliance Checks](./compliance-checks.md) - Regular compliance procedures

## Runbook Template

When creating new runbooks, use the following template:

```markdown
# [Runbook Title]

## Overview
Brief description of when to use this runbook.

## Prerequisites
- Required access levels
- Required tools
- Required knowledge

## Detection
How to detect this issue:
- Symptoms
- Alerts
- Monitoring indicators

## Investigation
Steps to investigate the issue:
1. Check X
2. Verify Y
3. Analyze Z

## Resolution
Step-by-step resolution:
1. Do A
2. Execute B
3. Verify C

## Post-Resolution
- Verification steps
- Communication requirements
- Follow-up actions

## Prevention
How to prevent this issue in the future.

## Related Links
- Monitoring dashboards
- Documentation
- Previous incidents
```

## Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | Slack: #oncall | Primary |
| Platform Team | Slack: #platform | Secondary |
| Security Team | Slack: #security | Security Issues |
| Management | Email/Phone | Critical Issues |

## Severity Levels

### Critical (P0)
- Service completely down
- Data loss or corruption
- Security breach
- **Response Time:** Immediate
- **Resolution Target:** 1 hour

### High (P1)
- Significant functionality impaired
- Performance severely degraded
- **Response Time:** 15 minutes
- **Resolution Target:** 4 hours

### Medium (P2)
- Minor functionality issues
- Performance slightly degraded
- **Response Time:** 1 hour
- **Resolution Target:** 24 hours

### Low (P3)
- Cosmetic issues
- Feature requests
- **Response Time:** Next business day
- **Resolution Target:** 1 week

## Quick Reference Commands

```bash
# Health check
curl -f http://localhost:8000/health

# Application logs
docker logs startup-studio-api --tail 100 -f

# Database connection
psql -h localhost -U studio -d studio

# Metrics endpoint
curl http://localhost:9102/metrics

# Container status
docker ps | grep studio

# System resources
htop
df -h
free -h
```

## Escalation Matrix

```
Issue Detected
    ↓
Initial Response (5 min)
    ↓
Investigation (15 min)
    ↓
Resolution Attempt (30 min)
    ↓
Escalate to Senior Engineer
    ↓
Escalate to Team Lead
    ↓
Escalate to Management
```

## Documentation Updates

Runbooks should be updated:
- After each incident
- When procedures change
- Quarterly review cycle
- When new features are deployed

---

*Last updated: 2025-07-27*  
*Next review: 2025-08-27*