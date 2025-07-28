# Monitoring & Observability Guide

## Overview

The Agentic Startup Studio includes comprehensive monitoring and observability capabilities to ensure system health, performance, and reliability in production environments.

## Architecture

```mermaid
graph TB
    App[Application] --> Metrics[Metrics Endpoint]
    App --> Logs[Structured Logs]
    App --> Traces[Distributed Traces]
    
    Metrics --> Prometheus[Prometheus]
    Logs --> Loki[Loki/ELK]
    Traces --> Jaeger[Jaeger]
    
    Prometheus --> Grafana[Grafana]
    Loki --> Grafana
    Jaeger --> Grafana
    
    Prometheus --> AlertManager[AlertManager]
    AlertManager --> Notifications[Slack/Email/PagerDuty]
```

## Components

### 1. Metrics Collection (Prometheus)

**Endpoint**: `http://localhost:9102/metrics`

**Key Metrics**:
- HTTP request duration and count
- Database query performance
- AI API usage and costs
- System resource utilization
- Business metrics (ideas processed, validation success rate)

**Configuration**: `monitoring/prometheus.yml`

### 2. Visualization (Grafana)

**URL**: `http://localhost:3000`
**Default Credentials**: `admin/admin`

**Dashboards**:
- System Overview
- Application Performance
- Business Metrics
- AI Usage & Costs
- Infrastructure Health

### 3. Alerting (AlertManager)

**Configuration**: `monitoring/alertmanager.yml`

**Alert Rules**:
- High error rate
- Response time degradation
- Database connection issues
- AI API rate limits
- Resource exhaustion

### 4. Distributed Tracing (OpenTelemetry)

**Collector**: OpenTelemetry Collector
**Backend**: Jaeger
**Frontend**: Jaeger UI

**Traced Operations**:
- HTTP requests
- Database queries
- AI API calls
- Inter-service communication

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f monitoring/docker-compose.yml up -d

# Verify services
docker ps | grep -E "(prometheus|grafana|alertmanager)"
```

### 2. Access Dashboards

```bash
# Grafana (main dashboard)
open http://localhost:3000

# Prometheus (metrics)
open http://localhost:9090

# AlertManager (alerts)
open http://localhost:9093
```

### 3. Test Metrics Collection

```bash
# Generate some test traffic
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Check if metrics are being collected
curl http://localhost:9090/api/v1/query?query=up
```

## Monitoring Configuration

### Application Metrics

The application exposes metrics at `/metrics` endpoint:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Business metrics
IDEAS_PROCESSED = Counter('ideas_processed_total', 'Total ideas processed')
AI_API_CALLS = Counter('ai_api_calls_total', 'AI API calls', ['provider', 'model'])
VALIDATION_SUCCESS_RATE = Gauge('validation_success_rate', 'Idea validation success rate')
```

### Custom Dashboards

Create custom Grafana dashboards by:

1. Accessing Grafana UI
2. Import dashboard JSON from `grafana/dashboards/`
3. Customize panels and queries
4. Export and save to version control

### Alert Configuration

Edit `monitoring/alerts.yml` to customize alerting rules:

```yaml
groups:
- name: application.rules
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: Error rate is {{ $value }} errors per second
```

## Observability Best Practices

### 1. Structured Logging

Use structured logging with consistent fields:

```python
import structlog

logger = structlog.get_logger()

logger.info("idea_processed", 
           idea_id=idea.id,
           user_id=user.id,
           duration=processing_time,
           status="success")
```

### 2. Distributed Tracing

Instrument key operations:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("process_idea")
def process_idea(idea_data):
    with tracer.start_as_current_span("validate_idea") as span:
        span.set_attribute("idea.category", idea_data.category)
        # Validation logic
        pass
```

### 3. Business Metrics

Track business-relevant metrics:

```python
# Track idea processing pipeline
IDEAS_IN_PIPELINE = Gauge('ideas_in_pipeline', 'Ideas currently in pipeline', ['stage'])
PROCESSING_TIME = Histogram('idea_processing_duration', 'Idea processing time', ['stage'])
COST_PER_IDEA = Histogram('cost_per_idea_dollars', 'Cost per idea validation')
```

## Production Monitoring Setup

### 1. External Monitoring

Configure external monitoring services:

- **Uptime**: UptimeRobot, Pingdom
- **APM**: New Relic, Datadog, Dynatrace
- **Error Tracking**: Sentry
- **Log Management**: ELK Stack, Splunk

### 2. SLI/SLO Configuration

Define Service Level Indicators and Objectives:

```yaml
slis:
  availability:
    description: "Percentage of successful HTTP requests"
    query: "rate(http_requests_total{status!~'5..'}[1m]) / rate(http_requests_total[1m])"
    target: 99.9%
  
  latency:
    description: "95th percentile response time"
    query: "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
    target: 200ms
  
  throughput:
    description: "Requests per second"
    query: "rate(http_requests_total[1m])"
    target: 100/sec
```

### 3. Runbook Integration

Link alerts to runbooks:

```yaml
- alert: DatabaseConnectionFailure
  annotations:
    runbook_url: https://docs.company.com/runbooks/database-connection-failure
    summary: Database connection pool exhausted
```

## Advanced Observability

### 1. Custom Metrics

Implement application-specific metrics:

```python
# AI model performance tracking
MODEL_INFERENCE_TIME = Histogram('ai_model_inference_seconds', 
                                 'AI model inference time', 
                                 ['model', 'provider'])

# Budget tracking
BUDGET_UTILIZATION = Gauge('budget_utilization_ratio', 
                          'Budget utilization ratio',
                          ['budget_type'])

# Quality metrics
IDEA_QUALITY_SCORE = Histogram('idea_quality_score', 
                               'Idea quality score distribution')
```

### 2. Distributed Tracing

Set up comprehensive tracing:

```python
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

# Add span processor
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### 3. Log Aggregation

Configure centralized logging:

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: app.logs
  
  fluentd:
    image: fluent/fluentd:v1.14
    volumes:
      - ./logging/fluent.conf:/fluentd/etc/fluent.conf
    ports:
      - "24224:24224"
```

## Monitoring Checklist

### Development
- [ ] Metrics endpoint accessible
- [ ] Basic dashboards configured
- [ ] Log levels appropriate
- [ ] Health checks implemented

### Staging
- [ ] Alert rules configured
- [ ] Dashboard alerts set up
- [ ] Load testing metrics collected
- [ ] Error tracking configured

### Production
- [ ] External monitoring configured
- [ ] SLO/SLI defined and measured
- [ ] Runbooks linked to alerts
- [ ] Log retention policies set
- [ ] Backup monitoring configured
- [ ] Security monitoring enabled

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Verify Prometheus configuration
docker logs prometheus

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### Grafana Dashboard Not Loading
```bash
# Check Grafana logs
docker logs grafana

# Verify datasource configuration
curl http://admin:admin@localhost:3000/api/datasources

# Check dashboard JSON syntax
```

#### Alerts Not Firing
```bash
# Check AlertManager configuration
curl http://localhost:9093/api/v1/status

# Verify alert rules
curl http://localhost:9090/api/v1/rules

# Check notification channels
```

### Performance Optimization

#### Prometheus
```yaml
# prometheus.yml optimizations
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  
scrape_configs:
- job_name: 'app'
  scrape_interval: 5s  # High-frequency for critical metrics
  metrics_path: /metrics
  static_configs:
  - targets: ['app:8000']
```

#### Grafana
```ini
# grafana.ini optimizations
[database]
type = sqlite3
cache_mode = private

[dashboards]
default_home_dashboard_path = /var/lib/grafana/dashboards/overview.json

[alerting]
enabled = true
execute_alerts = true
```

## Security Considerations

### 1. Access Control

```yaml
# Grafana LDAP configuration
[auth.ldap]
enabled = true
config_file = /etc/grafana/ldap.toml

# Prometheus authentication
basic_auth_users:
  admin: $2y$10$...
```

### 2. Network Security

```yaml
# Restrict access to monitoring ports
networks:
  monitoring:
    driver: bridge
    internal: true
```

### 3. Data Retention

```yaml
# Prometheus retention policy
command:
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=10GB'
```

## Integration Examples

### Slack Notifications

```yaml
# alertmanager.yml
route:
  receiver: 'slack-notifications'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### PagerDuty Integration

```yaml
receivers:
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .GroupLabels.instance }}'
```

For detailed configuration examples and advanced setup, see the individual configuration files in the `monitoring/` directory.