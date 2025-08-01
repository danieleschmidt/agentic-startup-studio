# API Outage Response Runbook

## Overview
Complete API outage affecting all endpoints. This is a **CRITICAL** incident requiring immediate response.

**Estimated Impact**: All users unable to access the platform
**Response Time**: 5 minutes
**Escalation**: Immediate PagerDuty alert + Slack #critical-alerts

## Symptoms
- API health check returning 5xx errors or timeouts
- Zero successful API requests in monitoring
- User reports of complete service unavailability
- Load balancer health checks failing

## Immediate Response (0-5 minutes)

### 1. Acknowledge and Assess
```bash
# Acknowledge the alert
curl -X POST "$PAGERDUTY_API/incidents/$INCIDENT_ID/acknowledge" \
  -H "Authorization: Token token=$PAGERDUTY_TOKEN"

# Quick status check
curl -f http://localhost:8000/health || echo "API DOWN"
curl -f http://localhost:8000/health/detailed || echo "DETAILED HEALTH FAILED"
```

### 2. Check System Resources
```bash
# Check if containers are running
docker ps | grep -E "(agentic|api|gateway)"

# Check system resources
free -h
df -h
top -bn1 | head -20
```

### 3. Initial Communication
Post in #critical-alerts:
```
🚨 CRITICAL: API Outage Detected
- Time: [TIMESTAMP]  
- Impact: Complete service unavailability
- Investigating: [YOUR NAME]
- ETA for update: 10 minutes
```

## Investigation (5-15 minutes)

### 4. Check Application Logs
```bash
# API server logs
docker logs agentic-api --tail=100 -f

# Database connectivity
docker logs postgres --tail=50

# Look for common issues
grep -i "error\|exception\|failed\|timeout" /var/log/agentic-studio/api.log | tail -20
```

### 5. Verify Dependencies
```bash
# Database connectivity
psql postgresql://studio:studio@localhost:5432/studio -c "SELECT 1"

# Redis connectivity  
redis-cli ping

# External API connectivity
curl -I https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 6. Check Recent Deployments
```bash
# Recent git changes
git log --oneline -10

# Recent container restarts
docker events --since="1h" | grep -E "(start|stop|die)"

# Check if deployment is in progress
kubectl get deployments -n production  # if using k8s
```

## Resolution Steps

### 7. Common Quick Fixes

#### A. Container Restart
```bash
# Restart API container
docker-compose restart api

# Wait and verify
sleep 30
curl -f http://localhost:8000/health
```

#### B. Database Connection Issues
```bash
# Restart database connection pool
docker-compose restart postgres

# Clear Redis cache
redis-cli flushall

# Restart API after database is ready
docker-compose restart api
```

#### C. Resource Exhaustion
```bash
# If memory issue
docker-compose restart api
docker system prune -f

# If disk space issue
df -h
# Clean up logs if needed
sudo journalctl --vacuum-time=1d
docker system prune -a -f
```

### 8. Rollback if Recent Deployment
```bash
# Rollback to previous version
git log --oneline -5
git checkout [PREVIOUS_STABLE_COMMIT]
./scripts/build.sh --tag rollback-$(date +%s)
docker-compose up -d api
```

## Verification (15-20 minutes)

### 9. Confirm Resolution
```bash
# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8000/health/detailed

# Test critical endpoints
curl -f http://localhost:8000/api/v1/ideas/validate \
  -H "Content-Type: application/json" \
  -d '{"idea": "test", "budget": 10}'

# Check monitoring dashboards
# - API response times
# - Error rates
# - Database connections
```

### 10. Update Communications
Post in #critical-alerts:
```
✅ RESOLVED: API Outage
- Resolution time: [DURATION]
- Root cause: [BRIEF CAUSE]
- Services restored: All API endpoints
- Monitoring: Normal metrics resumed
```

## Post-Incident (20+ minutes)

### 11. Documentation
- Log incident details in incident tracking system
- Update monitoring if gaps were identified
- Schedule post-mortem if outage > 15 minutes

### 12. Preventive Measures
- Review error logs for patterns
- Check if additional monitoring needed
- Verify backup/failover procedures worked
- Update runbooks based on lessons learned

## Escalation Procedures

### If Resolution Takes > 15 Minutes:
1. Escalate to Engineering Lead via PagerDuty
2. Consider activating incident commander
3. Prepare for external communication
4. Begin detailed incident timeline

### If Resolution Takes > 30 Minutes:
1. Activate full incident response team
2. Consider enabling maintenance mode
3. Prepare customer communication
4. Begin post-mortem preparation

## Common Root Causes

1. **Database Connection Pool Exhaustion**
   - Symptoms: Slow queries, connection timeouts
   - Solution: Restart API, optimize queries

2. **Memory Leak in API Process**
   - Symptoms: Gradual performance degradation
   - Solution: Container restart, investigate code

3. **External API Rate Limiting**
   - Symptoms: 429 errors in logs
   - Solution: Implement backoff, check quotas

4. **Resource Exhaustion (CPU/Memory/Disk)**
   - Symptoms: System-wide slowness
   - Solution: Scale resources, clean up

5. **Configuration Error from Deployment**
   - Symptoms: Immediate failure after deploy
   - Solution: Rollback deployment

## Related Documentation
- [Database Failure Runbook](database-failure.md)
- [High Error Rate Runbook](error-rate-spike.md) 
- [System Architecture](../../docs/ARCHITECTURE.md)
- [Deployment Procedures](../deployment/README.md)