# Incident Response Runbook
*Terragon Labs Agentic Startup Studio*

> **🚨 EMERGENCY CONTACT**: For P0/P1 incidents, immediately contact the on-call engineer via PagerDuty or Slack #emergency-response

## Quick Reference

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| **P0 - Critical** | 15 minutes | Immediate escalation to SRE team lead |
| **P1 - High** | 1 hour | Escalate to engineering manager within 2 hours |
| **P2 - Medium** | 4 hours | Escalate if not resolved within 8 hours |
| **P3 - Low** | 24 hours | Weekly review in engineering standup |

---

## Incident Classification

### P0 - Critical (Production Down)
- **Definition**: Complete service outage, data loss, or security breach
- **Examples**: 
  - API gateway completely unavailable
  - Database corruption or total failure
  - Security incident with data exposure
- **Response**: Immediate all-hands response, customer communication within 30 minutes

### P1 - High (Severe Impact)
- **Definition**: Major functionality broken, significant performance degradation
- **Examples**: 
  - Pipeline processing failures affecting >50% of requests
  - Authentication system intermittent failures
  - Critical AI model inference errors
- **Response**: Dedicated incident commander assigned, customer notification within 2 hours

### P2 - Medium (Partial Impact)
- **Definition**: Feature degradation, moderate performance issues
- **Examples**: 
  - Single service component failure with fallback working
  - Elevated error rates but system functional
  - Monitoring/alerting issues
- **Response**: Standard engineering response, internal notification

### P3 - Low (Minor Impact)
- **Definition**: Cosmetic issues, non-critical functionality affected
- **Examples**: 
  - Documentation inconsistencies
  - Non-critical log errors
  - Minor UI issues
- **Response**: Standard bug fix process

---

## Immediate Response Procedures

### Step 1: Initial Assessment (0-5 minutes)
```bash
# Quick health check commands
curl -f https://api.terragonlabs.com/health
python scripts/run_health_checks.py --critical-only
kubectl get pods -n production | grep -v Running
```

**Assessment Questions:**
- [ ] Is the main API responding?
- [ ] Are critical services (auth, pipeline, database) healthy?
- [ ] What is the blast radius (how many users affected)?
- [ ] Is this a known issue with existing runbook?

### Step 2: Incident Declaration (5-10 minutes)
```bash
# Declare incident in Slack
/incident declare severity=P1 title="API Gateway Outage" 

# Start incident bridge
/incident bridge

# Page on-call engineer (for P0/P1)
/pager alert "CRITICAL: Production outage requires immediate attention"
```

### Step 3: Initial Mitigation (10-30 minutes)
**For API/Service Issues:**
```bash
# Check recent deployments
git log --oneline --since="2 hours ago"
kubectl rollout history deployment/api-gateway -n production

# Quick rollback if recent deployment suspected
kubectl rollout undo deployment/api-gateway -n production

# Scale up replicas if capacity issue
kubectl scale deployment/api-gateway --replicas=10 -n production
```

**For Database Issues:**
```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Check active connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Check for long-running queries
psql $DATABASE_URL -c "SELECT query, query_start FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"
```

**For Pipeline Issues:**
```bash
# Check pipeline health
python -m pipeline.cli.ingestion_cli health

# Check message queue status
python -c "from pipeline.infrastructure.simple_health import check_queue_health; print(check_queue_health())"

# Restart pipeline workers
kubectl restart deployment/pipeline-worker -n production
```

---

## Detailed Investigation Playbooks

### API Gateway Outage
```bash
# 1. Check load balancer status
curl -I https://api.terragonlabs.com/health

# 2. Check container logs
kubectl logs -n production deployment/api-gateway --tail=100

# 3. Check resource utilization
kubectl top pods -n production -l app=api-gateway

# 4. Check ingress configuration
kubectl get ingress -n production -o yaml

# 5. Verify SSL certificate
openssl s_client -connect api.terragonlabs.com:443 -servername api.terragonlabs.com
```

### Database Performance Issues
```sql
-- Check for blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- Check table sizes and bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;

-- Kill problematic queries (use carefully)
-- SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '10 minutes' AND state = 'active';
```

### AI Pipeline Failures
```bash
# Check AI service status
python -c "
from pipeline.services.evidence_collector import EvidenceCollector
from pipeline.services.pitch_deck_generator import PitchDeckGenerator

try:
    collector = EvidenceCollector()
    result = collector.health_check()
    print(f'Evidence Collector: {result}')
except Exception as e:
    print(f'Evidence Collector Error: {e}')

try:
    generator = PitchDeckGenerator()
    result = generator.health_check()
    print(f'Pitch Generator: {result}')
except Exception as e:
    print(f'Pitch Generator Error: {e}')
"

# Check OpenAI API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Check vector database
python -c "
from pipeline.storage.idea_repository import IdeaRepository
repo = IdeaRepository()
try:
    result = repo.health_check()
    print(f'Vector DB Health: {result}')
except Exception as e:
    print(f'Vector DB Error: {e}')
"
```

---

## Communication Templates

### Initial Customer Communication (P0/P1)
```
🚨 INCIDENT ALERT - [TIMESTAMP]

We are currently investigating reports of [BRIEF_DESCRIPTION]. 
Our engineering team has been notified and is actively working on a resolution.

Affected services: [LIST_SERVICES]
Current status: [INVESTIGATING/IDENTIFIED/FIXING]
ETA for resolution: [TIME_ESTIMATE or "Under investigation"]

We will provide updates every 30 minutes until resolved.

Status page: https://status.terragonlabs.com
Incident ID: INC-[NUMBER]
```

### Resolution Communication
```
✅ RESOLVED - [TIMESTAMP]

The incident affecting [SERVICES] has been resolved.

Summary:
- Issue: [ROOT_CAUSE_SUMMARY]
- Duration: [TOTAL_TIME]
- Impact: [USER_IMPACT_DESCRIPTION]
- Resolution: [WHAT_WAS_DONE]

Next steps:
- Post-incident review scheduled for [DATE]
- Preventive measures being implemented
- Full post-mortem will be published within 48 hours

Thank you for your patience.
```

---

## Post-Incident Procedures

### Immediate (Within 24 hours)
1. **Incident Review Meeting**
   - Incident commander presents timeline
   - Technical leads explain root cause
   - Identify immediate action items

2. **Customer Follow-up**
   - Send detailed resolution communication
   - Offer service credits if applicable
   - Schedule customer success check-in

### Within 1 Week
1. **Post-Mortem Document**
   - Complete timeline of events
   - Root cause analysis
   - Action items with owners and deadlines
   - Process improvements identified

2. **Engineering Changes**
   - Implement immediate fixes
   - Update monitoring and alerting
   - Enhance runbooks based on learnings

### Monthly Review
1. **Incident Trends Analysis**
   - Review all incidents from past month
   - Identify patterns and systemic issues
   - Update escalation procedures if needed

---

## Emergency Contacts

| Role | Primary | Secondary |
|------|---------|-----------|
| **Incident Commander** | @alice.chen (Slack) | @bob.smith (Slack) |
| **Engineering Manager** | +1-555-0101 | +1-555-0102 |
| **SRE On-Call** | PagerDuty Auto | @sre-team (Slack) |
| **Security Team** | @security-team (Slack) | security@terragonlabs.com |
| **Executive Escalation** | @cto (Slack) | @ceo (Slack) |

---

## Useful Commands Reference

### Kubernetes
```bash
# Get all pod status
kubectl get pods -A | grep -v Running

# Check recent events
kubectl get events --sort-by=.metadata.creationTimestamp

# Get pod logs
kubectl logs -f deployment/api-gateway -n production

# Check resource usage
kubectl top nodes
kubectl top pods -A --sort-by=cpu
```

### Database
```bash
# Quick connection test
pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER

# Check active connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Database size
psql $DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size(current_database()));"
```

### Application Health
```bash
# Run health checks
python scripts/run_health_checks.py --verbose

# Check API endpoints
curl -f https://api.terragonlabs.com/health
curl -f https://api.terragonlabs.com/metrics

# Test pipeline functionality
python -m pipeline.cli.ingestion_cli health
```

---

*Last updated: $(date) | Review frequency: Monthly | Owner: SRE Team*