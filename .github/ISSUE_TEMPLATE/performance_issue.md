---
name: 🚀 Performance Issue
about: Report performance bottlenecks or optimization opportunities
title: '[PERF] '
labels: ['performance', 'optimization', 'needs-investigation']
assignees: ['terragonlabs/performance-team']
---

## Performance Issue Description

**Affected Component:**
- [ ] API Endpoints
- [ ] Database Queries
- [ ] Vector Search
- [ ] AI/ML Pipeline
- [ ] Frontend/UI
- [ ] Background Jobs
- [ ] Other: ___________

**Issue Summary:**
A clear and concise description of the performance issue.

## Performance Metrics

**Current Performance:**
- Response Time: _____ ms
- Throughput: _____ requests/second
- Memory Usage: _____ MB
- CPU Usage: _____ %
- Database Query Time: _____ ms

**Expected Performance:**
- Target Response Time: _____ ms
- Target Throughput: _____ requests/second
- Expected Memory Usage: _____ MB
- Expected CPU Usage: _____ %

## Reproduction Steps

1. Setup conditions: ...
2. Execute operation: ...
3. Measure performance: ...
4. Observe degradation: ...

## Environment Details

**System Configuration:**
- OS: ___________
- Python Version: ___________
- Hardware: ___________
- Docker/Container: [ ] Yes [ ] No
- Load Conditions: ___________

**Dependencies:**
- Database: PostgreSQL version _____
- Redis: version _____
- AI Model: ___________
- Other relevant versions: ___________

## Profiling Data

**Performance Profiles:** (attach profiling outputs)
- [ ] CPU profiler output
- [ ] Memory profiler output
- [ ] Database query analysis
- [ ] Network latency analysis

**Monitoring Screenshots:**
- [ ] Grafana dashboards
- [ ] Application logs
- [ ] System metrics

## Impact Assessment

**Business Impact:**
- [ ] Critical - System unusable
- [ ] High - Significant user experience degradation
- [ ] Medium - Noticeable but manageable
- [ ] Low - Minor optimization opportunity

**Affected Users:**
- Number of users affected: _____
- User segments affected: _____
- Peak usage times affected: _____

## Proposed Solutions

**Optimization Ideas:**
1. _____
2. _____
3. _____

**Alternative Approaches:**
- _____
- _____

## Additional Context

**Related Issues:**
- #_____ (link to related performance issues)
- #_____ (link to related feature requests)

**Supporting Documentation:**
- [ ] Performance benchmarks
- [ ] Load testing results
- [ ] Architecture diagrams
- [ ] Code samples

**Priority Justification:**
Explain why this performance issue should be prioritized.

---

### For Maintainers

**Performance Review Checklist:**
- [ ] Verified performance metrics
- [ ] Reproduced the issue locally
- [ ] Analyzed profiling data
- [ ] Identified root cause
- [ ] Estimated effort for optimization
- [ ] Assigned appropriate priority label
- [ ] Added to performance backlog