# Data Pipeline Operations Manual

This manual provides comprehensive guidance for operating and maintaining the Agentic Startup Studio data pipeline in production environments. It covers monitoring, budget management, performance optimization, and incident response procedures.

## Table of Contents

- [System Overview](#system-overview)
- [Daily Operations](#daily-operations)
- [Budget Monitoring and Cost Management](#budget-monitoring-and-cost-management)
- [System Health Monitoring](#system-health-monitoring)
- [Performance Monitoring and Optimization](#performance-monitoring-and-optimization)
- [Error Handling and Recovery](#error-handling-and-recovery)
- [Maintenance Procedures](#maintenance-procedures)
- [Incident Response](#incident-response)
- [Troubleshooting](#troubleshooting)
- [Backup and Recovery](#backup-and-recovery)

---

## System Overview

### Architecture Components

**Core Services:**
- **Data Ingestion Pipeline:** Idea submission and validation
- **Budget Sentinel:** Real-time cost tracking (≤$62/cycle)
- **Workflow Orchestrator:** Multi-stage pipeline coordination
- **Evidence Collector:** RAG-based research gathering
- **Pitch Deck Generator:** Automated presentation creation
- **Campaign Generator:** Google Ads automation

**External Dependencies:**
- **PostgreSQL + pgvector:** Data storage and similarity search
- **OpenAI API:** Text processing and generation
- **Google Ads API:** Campaign management
- **PostHog:** Analytics and tracking

### Service States

| State | Description | Actions Required |
|-------|-------------|------------------|
| `HEALTHY` | All systems operational | Normal monitoring |
| `WARNING` | Non-critical issues detected | Increased monitoring |
| `CRITICAL` | Service degradation | Immediate investigation |
| `EMERGENCY` | System failure or budget exceeded | Emergency response |

---

## Daily Operations

### Morning Health Check (15 minutes)

```bash
# 1. Check system status
sudo systemctl status agentic-pipeline

# 2. Review budget status
python -m pipeline.cli.ingestion_cli budget status

# 3. Check database connectivity
python -m pipeline.cli.ingestion_cli db status

# 4. Review overnight logs
tail -50 logs/pipeline.log | grep -E "(ERROR|CRITICAL|WARNING)"

# 5. Verify external service connectivity
python -c "
from pipeline.config.settings import get_settings
config = get_settings()
print(f'Environment: {config.environment}')
print(f'Debug mode: {config.debug_mode}')
"
```

### Daily Budget Review

```bash
# Check current cycle budget status
python -m pipeline.cli.ingestion_cli budget status --detailed

# Review spending breakdown by category
python -c "
from pipeline.services.budget_sentinel import get_budget_sentinel
sentinel = get_budget_sentinel()
status = sentinel.get_budget_status()

print('=== DAILY BUDGET REPORT ===')
print(f'Cycle: {status[\"cycle_id\"]}')
print(f'Total Budget: ${status[\"total_budget\"][\"allocated\"]:.2f}')
print(f'Total Spent: ${status[\"total_budget\"][\"spent\"]:.2f}')
print(f'Remaining: ${status[\"total_budget\"][\"remaining\"]:.2f}')
print(f'Usage: {status[\"total_budget\"][\"usage_percentage\"]:.1f}%')

print('\n=== BY CATEGORY ===')
for cat, data in status['categories'].items():
    print(f'{cat}: ${data[\"spent\"]:.2f}/${data[\"allocated\"]:.2f} ({data[\"usage_percentage\"]:.1f}%)')
"
```

### Pipeline Activity Monitoring

```bash
# Check active ideas in pipeline
python -m pipeline.cli.ingestion_cli list --status=processing

# Monitor pipeline stage distribution
python -c "
from pipeline.storage.idea_repository import IdeaRepository
repo = IdeaRepository()

stages = {}
for idea in repo.get_all_ideas():
    stage = idea.current_stage
    stages[stage] = stages.get(stage, 0) + 1

print('=== PIPELINE STAGE DISTRIBUTION ===')
for stage, count in stages.items():
    print(f'{stage}: {count} ideas')
"
```

---

## Budget Monitoring and Cost Management

### Budget Thresholds and Alerts

**Automatic Alert Levels:**
- **Warning (80%):** Email notification to operators
- **Critical (95%):** Slack alert + throttling of non-critical operations
- **Emergency (100%):** Immediate shutdown + executive notification

### Budget Monitoring Commands

```bash
# Real-time budget tracking
watch -n 30 'python -m pipeline.cli.ingestion_cli budget status'

# Historical spending analysis
python -c "
from pipeline.services.budget_sentinel import get_budget_sentinel
import json

sentinel = get_budget_sentinel()
status = sentinel.get_budget_status()

# Export for analysis
with open('budget_report.json', 'w') as f:
    json.dump(status, f, indent=2)

print('Budget report exported to budget_report.json')
"

# Check for budget violations
python -c "
from pipeline.services.budget_sentinel import get_budget_sentinel

sentinel = get_budget_sentinel()
if sentinel.emergency_shutdown:
    print('⚠️  EMERGENCY SHUTDOWN ACTIVE')
    exit(1)
elif sentinel.circuit_breaker_active:
    print('⚠️  CIRCUIT BREAKER ACTIVE')
    exit(1)
else:
    print('✅ Budget systems operational')
"
```

### Cost Optimization Strategies

**1. OpenAI Token Management**
- Monitor token usage per operation
- Implement response caching for repeated queries
- Use cheaper models for non-critical operations

**2. Google Ads Optimization**
- Set daily spending limits per campaign
- Monitor CTR and conversion rates
- Pause underperforming campaigns automatically

**3. Infrastructure Cost Control**
- Schedule non-critical jobs during off-peak hours
- Implement auto-scaling based on load
- Regular cleanup of temporary files and logs

### Budget Cycle Management

```bash
# Start new budget cycle (use with caution)
python -m pipeline.cli.ingestion_cli budget reset --confirm

# Archive previous cycle data
python -c "
from datetime import datetime
import shutil

# Create archive directory
archive_dir = f'archives/budget_cycle_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}'
shutil.copytree('logs', f'{archive_dir}/logs')
print(f'Logs archived to {archive_dir}')
"
```

---

## System Health Monitoring

### Core System Metrics

**1. Application Health**
```bash
# Check service status
systemctl is-active agentic-pipeline

# Monitor memory usage
ps aux | grep python | grep pipeline

# Check disk space
df -h /opt/agentic-startup-studio

# Monitor database connections
sudo -u postgres psql -c "
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';
"
```

**2. Database Health**
```bash
# Check database performance
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
ORDER BY n_tup_ins DESC;
"

# Monitor query performance
sudo -u postgres psql startup_studio_prod -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements 
WHERE calls > 10
ORDER BY total_time DESC 
LIMIT 10;
"

# Check vector index performance
sudo -u postgres psql startup_studio_prod -c "
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%vector%';
"
```

**3. External Service Health**
```bash
# Test OpenAI API connectivity
python -c "
import openai
import os
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    models = openai.Model.list()
    print('✅ OpenAI API: Connected')
except Exception as e:
    print(f'❌ OpenAI API: {e}')
"

# Test Google Ads API
python -c "
# Add Google Ads API test here when available
print('🔄 Google Ads API: Test implementation needed')
"

# Test PostHog connectivity
python -c "
import posthog
import os
try:
    posthog.project_api_key = os.getenv('POSTHOG_PROJECT_KEY')
    posthog.capture('health_check', {'test': True})
    print('✅ PostHog: Connected')
except Exception as e:
    print(f'❌ PostHog: {e}')
"
```

### Automated Health Checks

**Create health check script:** `/opt/agentic-startup-studio/scripts/health_check.sh`

```bash
#!/bin/bash
# System health check script

echo "=== AGENTIC STARTUP STUDIO HEALTH CHECK ==="
echo "Timestamp: $(date)"

# Check service status
if systemctl is-active --quiet agentic-pipeline; then
    echo "✅ Service: Running"
else
    echo "❌ Service: Stopped"
    exit 1
fi

# Check budget status
BUDGET_STATUS=$(python -m pipeline.cli.ingestion_cli budget status --json)
EMERGENCY=$(echo $BUDGET_STATUS | jq -r '.emergency_shutdown')

if [ "$EMERGENCY" = "true" ]; then
    echo "❌ Budget: Emergency shutdown active"
    exit 1
else
    echo "✅ Budget: Operational"
fi

# Check database
if python -m pipeline.cli.ingestion_cli db status >/dev/null 2>&1; then
    echo "✅ Database: Connected"
else
    echo "❌ Database: Connection failed"
    exit 1
fi

echo "✅ All systems healthy"
```

**Schedule health checks:**
```bash
# Add to crontab
echo "*/5 * * * * /opt/agentic-startup-studio/scripts/health_check.sh >> /var/log/health_check.log 2>&1" | crontab -
```

---

## Performance Monitoring and Optimization

### Key Performance Indicators (KPIs)

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Pipeline completion time | <4 hours | >4 hours | >6 hours |
| Database query time | <100ms | >500ms | >1000ms |
| API response time | <2s | >5s | >10s |
| Memory usage | <70% | >80% | >90% |
| Disk usage | <70% | >80% | >90% |

### Performance Monitoring Commands

```bash
# Monitor pipeline execution times
python -c "
from pipeline.storage.idea_repository import IdeaRepository
from datetime import datetime, timedelta

repo = IdeaRepository()
recent_ideas = repo.get_ideas_by_date_range(
    start_date=datetime.utcnow() - timedelta(days=7),
    end_date=datetime.utcnow()
)

print('=== PIPELINE PERFORMANCE (Last 7 days) ===')
total_time = 0
completed = 0

for idea in recent_ideas:
    if idea.status == 'completed':
        # Calculate completion time (implementation needed)
        print(f'Idea {idea.idea_id}: Completed')
        completed += 1

print(f'Completed ideas: {completed}')
print(f'Average completion time: Implementation needed')
"

# Monitor system resources
iostat -x 1 5  # I/O statistics
vmstat 1 5     # Memory and CPU
netstat -tuln  # Network connections
```

### Database Performance Optimization

```bash
# Analyze slow queries
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    query,
    calls,
    total_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 5;
"

# Check index usage
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
"

# Vacuum and analyze (run during maintenance windows)
sudo -u postgres psql startup_studio_prod -c "VACUUM ANALYZE;"
```

### Cache Performance Monitoring

```bash
# Monitor embedding cache performance
python -c "
from pipeline.config.settings import get_settings
config = get_settings()

if config.embedding.enable_cache:
    print('✅ Embedding cache: Enabled')
    print(f'Cache TTL: {config.embedding.cache_ttl}s')
    print(f'Cache size limit: {config.embedding.cache_size}')
else:
    print('❌ Embedding cache: Disabled')
"
```

---

## Error Handling and Recovery

### Common Error Scenarios

**1. Database Connection Issues**
```bash
# Check database status
sudo systemctl status postgresql

# Restart database service
sudo systemctl restart postgresql

# Test connection
python -m pipeline.cli.ingestion_cli db status

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

**2. External API Failures**
```bash
# OpenAI API issues
python -c "
import openai
try:
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt='Test',
        max_tokens=10
    )
    print('✅ OpenAI API working')
except openai.error.RateLimitError:
    print('❌ OpenAI: Rate limit exceeded')
except openai.error.InvalidRequestError as e:
    print(f'❌ OpenAI: Invalid request - {e}')
except Exception as e:
    print(f'❌ OpenAI: {e}')
"

# Check API key validity
python -c "
import os
key = os.getenv('OPENAI_API_KEY')
if not key:
    print('❌ OPENAI_API_KEY not set')
elif len(key) < 20:
    print('❌ OPENAI_API_KEY appears invalid')
else:
    print('✅ OPENAI_API_KEY format valid')
"
```

**3. Budget Sentinel Recovery**
```bash
# Reset emergency shutdown (use carefully)
python -c "
from pipeline.services.budget_sentinel import get_budget_sentinel

sentinel = get_budget_sentinel()
if sentinel.emergency_shutdown:
    print('⚠️  WARNING: Resetting emergency shutdown')
    print('This should only be done after addressing budget issues')
    
    # Uncomment next line only after verification
    # sentinel.emergency_shutdown = False
    # sentinel.circuit_breaker_active = False
    # print('✅ Emergency shutdown reset')
else:
    print('✅ No emergency shutdown active')
"
```

### Recovery Procedures

**1. Pipeline Stuck in Processing**
```bash
# Identify stuck ideas
python -c "
from pipeline.storage.idea_repository import IdeaRepository
from datetime import datetime, timedelta

repo = IdeaRepository()
cutoff = datetime.utcnow() - timedelta(hours=6)

stuck_ideas = []
for idea in repo.get_all_ideas():
    if idea.status == 'processing' and idea.updated_at < cutoff:
        stuck_ideas.append(idea)

print(f'Found {len(stuck_ideas)} potentially stuck ideas')
for idea in stuck_ideas:
    print(f'- {idea.idea_id}: {idea.current_stage} since {idea.updated_at}')
"

# Reset stuck idea (manual intervention)
python -m pipeline.cli.ingestion_cli reset-idea --id=<idea_id> --stage=<stage>
```

**2. Database Recovery**
```bash
# Create database backup before recovery
sudo -u postgres pg_dump startup_studio_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Check database integrity
sudo -u postgres psql startup_studio_prod -c "
SELECT pg_size_pretty(pg_database_size('startup_studio_prod')) as db_size;
"

# Repair corrupted indexes
sudo -u postgres psql startup_studio_prod -c "REINDEX DATABASE startup_studio_prod;"
```

---

## Maintenance Procedures

### Weekly Maintenance Tasks

**1. Log Rotation and Cleanup**
```bash
# Rotate application logs
sudo logrotate -f /etc/logrotate.d/agentic-pipeline

# Clean old log files (>30 days)
find logs/ -name "*.log.*" -mtime +30 -delete

# Archive old data
python -c "
from datetime import datetime, timedelta
import shutil

cutoff = datetime.utcnow() - timedelta(days=30)
print(f'Archiving data older than {cutoff}')
# Implementation: Move old ideas to archive table
"
```

**2. Database Maintenance**
```bash
# Vacuum and analyze database
sudo -u postgres psql startup_studio_prod -c "
VACUUM (VERBOSE, ANALYZE);
"

# Update database statistics
sudo -u postgres psql startup_studio_prod -c "
ANALYZE;
"

# Check and optimize indexes
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes 
ORDER BY pg_relation_size(indexrelid) DESC;
"
```

**3. Security Updates**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python dependencies
source venv/bin/activate
pip list --outdated
pip install -r requirements.txt --upgrade

# Check for security vulnerabilities
pip audit
```

### Monthly Maintenance Tasks

**1. Performance Review**
```bash
# Generate performance report
python scripts/generate_performance_report.py --days=30

# Review budget trends
python -c "
# Generate monthly budget analysis
print('Monthly budget analysis - implementation needed')
"

# Database performance analysis
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time
FROM pg_stat_statements 
WHERE calls > 100
ORDER BY total_time DESC 
LIMIT 20;
"
```

**2. Capacity Planning**
```bash
# Check disk space trends
df -h /opt/agentic-startup-studio
du -sh logs/ data/ backups/

# Monitor database growth
sudo -u postgres psql startup_studio_prod -c "
SELECT 
    pg_size_pretty(pg_total_relation_size('ideas')) as ideas_size,
    pg_size_pretty(pg_total_relation_size('evidence')) as evidence_size;
"

# Review resource utilization
sar -u 1 1  # CPU usage
sar -r 1 1  # Memory usage
sar -d 1 1  # Disk I/O
```

---

## Incident Response

### Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P1 - Critical** | System down, budget exceeded | 15 minutes | Immediate |
| **P2 - High** | Major functionality impaired | 1 hour | Within 30 min |
| **P3 - Medium** | Minor functionality impaired | 4 hours | Within 2 hours |
| **P4 - Low** | Cosmetic or documentation | Next business day | None |

### Emergency Contacts

```bash
# Emergency notification script
cat > /opt/agentic-startup-studio/scripts/emergency_notify.sh << 'EOF'
#!/bin/bash
INCIDENT_TYPE=$1
MESSAGE=$2

echo "EMERGENCY: $INCIDENT_TYPE - $MESSAGE" >> /var/log/emergency.log

# Add notification integrations:
# - Slack webhook
# - Email alerts
# - SMS notifications
# - PagerDuty integration

echo "Emergency notification sent: $INCIDENT_TYPE"
EOF

chmod +x /opt/agentic-startup-studio/scripts/emergency_notify.sh
```

### Critical Incident Procedures

**1. Budget Emergency Shutdown**
```bash
# Immediate response
echo "=== BUDGET EMERGENCY RESPONSE ==="
echo "1. Verify emergency shutdown status"
python -c "
from pipeline.services.budget_sentinel import get_budget_sentinel
sentinel = get_budget_sentinel()
print(f'Emergency shutdown: {sentinel.emergency_shutdown}')
print(f'Budget status: {sentinel.get_budget_status()}')
"

echo "2. Stop all non-critical operations"
# Implementation: Stop background tasks

echo "3. Notify stakeholders"
/opt/agentic-startup-studio/scripts/emergency_notify.sh "BUDGET_EXCEEDED" "Emergency shutdown activated"
```

**2. Database Failure**
```bash
# Database emergency response
echo "=== DATABASE EMERGENCY RESPONSE ==="
echo "1. Check database status"
sudo systemctl status postgresql

echo "2. Attempt restart"
sudo systemctl restart postgresql

echo "3. Check data integrity"
sudo -u postgres psql startup_studio_prod -c "SELECT 1;"

echo "4. Restore from backup if needed"
# Implementation: Backup restoration procedure
```

---

## Troubleshooting

### Common Issues and Solutions

**Issue: High Memory Usage**
```bash
# Identify memory-heavy processes
ps aux --sort=-%mem | head -20

# Check for memory leaks
valgrind --tool=massif python -m pipeline.main_pipeline

# Solution: Restart service
sudo systemctl restart agentic-pipeline
```

**Issue: Slow Database Queries**
```bash
# Identify slow queries
sudo -u postgres psql startup_studio_prod -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;
"

# Solution: Add indexes or optimize queries
```

**Issue: API Rate Limiting**
```bash
# Check rate limit status
python -c "
# Implementation: Check current API usage rates
print('API rate limit check - implementation needed')
"

# Solution: Implement exponential backoff
```

### Log Analysis Commands

```bash
# Search for errors in logs
grep -E "(ERROR|CRITICAL)" logs/pipeline.log | tail -20

# Monitor logs in real-time
tail -f logs/pipeline.log | grep -E "(ERROR|WARNING|CRITICAL)"

# Analyze error patterns
awk '/ERROR/ {print $4}' logs/pipeline.log | sort | uniq -c | sort -nr
```

---

For additional support, refer to:
- [Deployment Guide](deployment-guide.md) for installation issues
- [User Guide](user-guide.md) for operational workflows
- [API Documentation](api-documentation.md) for service interfaces

**Emergency Hotline:** Contact system administrator immediately for P1 incidents.