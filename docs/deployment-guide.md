# Data Pipeline Deployment Guide

This guide provides step-by-step instructions for deploying the Agentic Startup Studio data pipeline to production environments. The system implements a multi-stage validation workflow with automated budget controls and quality gates.

## Table of Contents

- [System Overview](#system-overview)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Security Configuration](#security-configuration)
- [Database Setup](#database-setup)
- [External Service Configuration](#external-service-configuration)
- [Application Deployment](#application-deployment)
- [Verification and Testing](#verification-and-testing)
- [Troubleshooting](#troubleshooting)

---

## System Overview

The data pipeline processes startup ideas through a structured workflow:

**Pipeline Stages:** `Ideate → Research → Deck → Investors → SmokeTest → MVP`

**Key Features:**
- Automated validation with 100% CI pass rate requirement
- Budget control: ≤$62 per cycle ($12 GPT + $50 ads)
- Quality gates with configurable thresholds
- PostgreSQL with pgvector for similarity search
- Multi-agent investor evaluation system
- Automated smoke test deployment

---

## System Requirements

### Hardware Requirements

**Minimum Production Setup:**
- **CPU:** 4 cores (8 vCPUs recommended)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 100GB SSD (500GB recommended for logs/data)
- **Network:** Stable internet connection with low latency

**Database Server (if separate):**
- **CPU:** 2 cores minimum (4 recommended)
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 50GB SSD with backup capacity

### Software Requirements

**Operating System:**
- Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- RHEL 8+ or CentOS Stream 8+
- macOS 12+ (development only)

**Required Software:**
- Python 3.9+ (3.11 recommended)
- PostgreSQL 14+ with pgvector extension
- Redis 6+ (for caching and queues)
- Docker 20.10+ and Docker Compose 2.0+
- Git 2.30+

**Optional but Recommended:**
- Nginx (reverse proxy)
- Let's Encrypt (SSL certificates)
- Grafana (monitoring dashboards)

---

## Environment Setup

### 1. Creating the Production Environment

```bash
# Create application user
sudo adduser --system --group --home /opt/agentic-startup-studio agentic

# Create application directory
sudo mkdir -p /opt/agentic-startup-studio
sudo chown agentic:agentic /opt/agentic-startup-studio

# Switch to application user
sudo -u agentic -i
cd /opt/agentic-startup-studio

# Clone repository
git clone https://github.com/your-org/agentic-startup-studio.git .
git checkout main  # or specific release tag
```

### 2. Python Environment Setup

```bash
# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import pipeline; print('Installation successful')"
```

### 3. Directory Structure

```bash
# Create required directories
mkdir -p {logs,data,backups,uploads,static}
mkdir -p data/{ideas,reports,campaigns}

# Set permissions
chmod 755 logs data backups
chmod 700 uploads  # More restrictive for uploaded content
```

---

## Security Configuration

### 1. Environment Variables

Create `/opt/agentic-startup-studio/.env` with production values:

```bash
# Environment
ENVIRONMENT=production
DEBUG_MODE=false
SECRET_KEY=your-256-bit-secret-key-here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=startup_studio_prod
DB_USER=agentic_user
DB_PASSWORD=your-secure-database-password

# External Service APIs
OPENAI_API_KEY=your-openai-api-key
GOOGLE_ADS_API_KEY=your-google-ads-key
POSTHOG_PROJECT_KEY=your-posthog-key

# Budget Controls
TOTAL_CYCLE_BUDGET=62.00
OPENAI_BUDGET=12.00
GOOGLE_ADS_BUDGET=45.00
INFRASTRUCTURE_BUDGET=5.00

# Security Settings
ALLOWED_ORIGINS=your-domain.com,api.your-domain.com
ENABLE_PROFANITY_FILTER=true
ENABLE_SPAM_DETECTION=true

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_JSON_LOGGING=true
```

### 2. File Permissions

```bash
# Secure environment file
chmod 600 .env
chown agentic:agentic .env

# Secure application files
find . -type f -name "*.py" -exec chmod 644 {} \;
find . -type d -exec chmod 755 {} \;

# Make CLI executable
chmod +x pipeline/cli/ingestion_cli.py
```

### 3. Firewall Configuration

```bash
# Ubuntu/Debian UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow from localhost to any port 5432  # PostgreSQL (internal)
sudo ufw enable

# RHEL/CentOS Firewalld
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## Database Setup

### 1. PostgreSQL Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-14 postgresql-client-14 postgresql-contrib-14

# RHEL/CentOS
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql
```

### 2. pgvector Extension

```bash
# Install pgvector extension
sudo apt install postgresql-14-pgvector  # Ubuntu/Debian

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. Database Setup

```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE startup_studio_prod;
CREATE USER agentic_user WITH ENCRYPTED PASSWORD 'your-secure-password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE startup_studio_prod TO agentic_user;
ALTER USER agentic_user CREATEDB;  -- For running migrations

-- Enable pgvector extension
\c startup_studio_prod
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify setup
SELECT * FROM pg_extension WHERE extname = 'vector';

-- PERFORMANCE OPTIMIZATION (PERF-002 Compliance)
-- Configure PostgreSQL for optimal vector search performance
SET shared_preload_libraries = 'pg_stat_statements';
SET effective_cache_size = '2GB';  -- Adjust based on available RAM
SET maintenance_work_mem = '512MB';  -- For index building
SET work_mem = '64MB';  -- For query operations
```

### 4. Run Database Migrations

```bash
# Activate virtual environment
source venv/bin/activate

# Run database initialization
python -m pipeline.cli.ingestion_cli db init

# Verify database schema
python -m pipeline.cli.ingestion_cli db status
```

---

## External Service Configuration

### 1. OpenAI API Setup

```bash
# Test API connection
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
print('OpenAI API connection successful')
"
```

### 2. Google Ads API Setup

1. **Google Cloud Console Setup:**
   - Create a new project or select existing
   - Enable Google Ads API
   - Create service account and download JSON key
   - Set up OAuth2 credentials for user access

2. **API Configuration:**
```bash
# Store Google Ads credentials securely
echo "your-google-ads-api-key" > /opt/agentic-startup-studio/config/google-ads-key.txt
chmod 600 /opt/agentic-startup-studio/config/google-ads-key.txt
```

### 3. PostHog Analytics Setup

```bash
# Test PostHog connection
python -c "
import posthog
posthog.project_api_key = os.getenv('POSTHOG_PROJECT_KEY')
posthog.host = 'https://app.posthog.com'
print('PostHog connection successful')
"
```

---

## Application Deployment

### 1. Systemd Service Configuration

Create `/etc/systemd/system/agentic-pipeline.service`:

```ini
[Unit]
Description=Agentic Startup Studio Pipeline
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=notify
User=agentic
Group=agentic
WorkingDirectory=/opt/agentic-startup-studio
Environment=PATH=/opt/agentic-startup-studio/venv/bin
ExecStart=/opt/agentic-startup-studio/venv/bin/python -m pipeline.main_pipeline
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/agentic-startup-studio
NoNewPrivileges=true
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Services

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable agentic-pipeline

# Start services
sudo systemctl start agentic-pipeline

# Check status
sudo systemctl status agentic-pipeline
```

### 3. Nginx Reverse Proxy (Optional)

Create `/etc/nginx/sites-available/agentic-pipeline`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # API proxy
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static/ {
        alias /opt/agentic-startup-studio/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

---

## Verification and Testing

### 1. Basic System Health Check

```bash
# Test database connectivity
python -m pipeline.cli.ingestion_cli db status

# Test configuration loading
python -c "
from pipeline.config.settings import get_settings
config = get_settings()
print(f'Environment: {config.environment}')
print(f'Database: {config.database.get_safe_connection_url()}')
"

# Test CLI interface
python -m pipeline.cli.ingestion_cli --help
```

### 2. Integration Tests

```bash
# Run integration test suite
python -m pytest tests/integration/ -v

# Test specific components
python -m pytest tests/pipeline/ingestion/ -v
python -m pytest tests/pipeline/storage/ -v
```

### 3. End-to-End Pipeline Test

```bash
# Submit test idea
python -m pipeline.cli.ingestion_cli submit \
  --title "Test Startup Idea" \
  --description "This is a test idea for deployment verification" \
  --category "technology"

# Monitor pipeline execution
python -m pipeline.cli.ingestion_cli status

# Check logs
tail -f logs/pipeline.log
```

### 4. Performance Validation (PERF-002 Compliance)

```bash
# Run PERF-002 performance validation  
python scripts/perf_002_validation.py --samples 100 --threshold 50.0

# Expected output should show:
# ✅ All queries under 50ms threshold
# ✅ PERF-002 COMPLIANT status

# Monitor vector search performance in production
python -c "
from pipeline.storage.optimized_vector_search import get_vector_search
import asyncio

async def check_performance():
    search = await get_vector_search()
    report = search.get_performance_report()
    print(f'PERF-002 Status: {report[\"perf_002_compliance\"][\"status\"]}')
    print(f'Average Query Time: {report[\"perf_002_compliance\"][\"current_avg_ms\"]:.2f}ms')
    print(f'Target: <{report[\"perf_002_compliance\"][\"target_ms\"]}ms')

asyncio.run(check_performance())
"

# Test concurrent load performance
python -c "
import asyncio
import time
from pipeline.storage.optimized_vector_search import get_vector_search

async def load_test():
    search = await get_vector_search()
    
    async def single_query():
        start = time.perf_counter()
        await search.similarity_search('test query', limit=10)
        return (time.perf_counter() - start) * 1000
    
    # Run 10 concurrent queries
    times = await asyncio.gather(*[single_query() for _ in range(10)])
    avg_time = sum(times) / len(times)
    
    print(f'Concurrent Load Test Results:')
    print(f'Average: {avg_time:.2f}ms')
    print(f'Max: {max(times):.2f}ms')
    print(f'PERF-002 Compliant: {max(times) < 50.0}')

asyncio.run(load_test())
"
```

### 4. Budget Monitoring Test

```bash
# Check budget tracking
python -c "
from pipeline.services.budget_sentinel import BudgetSentinel
sentinel = BudgetSentinel()
print(f'Current spending: ${sentinel.get_current_spending():.2f}')
print(f'Budget remaining: ${sentinel.get_remaining_budget():.2f}')
"
```

---

## Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection manually
psql "postgresql://agentic_user:password@localhost:5432/startup_studio_prod"

# Check firewall rules
sudo netstat -tlnp | grep 5432
```

**2. API Key Issues**
```bash
# Verify environment variables are loaded
python -c "import os; print('OPENAI_API_KEY' in os.environ)"

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

**3. Permission Problems**
```bash
# Fix file ownership
sudo chown -R agentic:agentic /opt/agentic-startup-studio

# Fix directory permissions
find /opt/agentic-startup-studio -type d -exec chmod 755 {} \;
find /opt/agentic-startup-studio -type f -exec chmod 644 {} \;
```

**4. Budget Sentinel Alerts**
```bash
# Check budget status
python -m pipeline.cli.ingestion_cli budget status

# Reset budget tracking (use carefully)
python -m pipeline.cli.ingestion_cli budget reset --confirm
```

### Log Analysis

```bash
# Application logs
tail -f logs/pipeline.log

# System logs
sudo journalctl -u agentic-pipeline -f

# Database logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log

# Nginx logs (if using reverse proxy)
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

```bash
# Check resource usage
htop
iostat -x 1
free -h

# Database performance
sudo -u postgres psql startup_studio_prod -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"
```

---

## Next Steps

After successful deployment:

1. **Review [Operations Manual](operations-manual.md)** for ongoing maintenance
2. **Set up monitoring dashboards** using provided Grafana configurations
3. **Configure backup procedures** for data protection
4. **Review [User Guide](user-guide.md)** for operational workflows
5. **Test disaster recovery procedures** in a safe environment

For additional support, consult the [API Documentation](api-documentation.md) and [Troubleshooting Guide](operations-manual.md#troubleshooting).