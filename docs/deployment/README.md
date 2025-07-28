# Deployment Guide - Agentic Startup Studio

This directory contains comprehensive deployment documentation and configuration for the Agentic Startup Studio.

## Quick Start

```bash
# Build production image
python scripts/build.py --target production

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d

# Check deployment health
curl http://localhost:8000/health
```

## Deployment Options

### 1. Docker Compose (Recommended for Development/Staging)

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Development deployment with hot reload
docker-compose -f docker-compose.dev.yml up -d

# Scaling services
docker-compose up -d --scale app=3
```

### 2. Kubernetes (Recommended for Production)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=agentic-startup-studio

# Access logs
kubectl logs -l app=agentic-startup-studio -f
```

### 3. Cloud Platforms

#### Google Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/agentic-startup-studio

# Deploy to Cloud Run
gcloud run deploy agentic-startup-studio \
  --image gcr.io/PROJECT_ID/agentic-startup-studio \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS ECS/Fargate
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag agentic-startup-studio:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/agentic-startup-studio:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/agentic-startup-studio:latest

# Deploy with ECS CLI or CDK
```

## Environment Variables

### Required Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# AI Services
OPENAI_API_KEY=your_openai_key
GOOGLE_AI_API_KEY=your_google_ai_key

# Security
SECRET_KEY=your_secret_key_min_32_chars
JWT_SECRET_KEY=your_jwt_secret
```

### Optional Variables
```bash
# Performance
WORKERS=4
MAX_REQUESTS=1000
TIMEOUT=30

# Features
ENABLE_TRACING=true
ENABLE_METRICS=true
LOG_LEVEL=INFO

# External Services
REDIS_URL=redis://localhost:6379
SENTRY_DSN=your_sentry_dsn
```

## Security Configuration

### 1. Secrets Management

#### Using Environment Variables (Development)
```bash
cp .env.example .env
# Edit .env with your values
```

#### Using Docker Secrets (Production)
```bash
echo "your_secret_value" | docker secret create db_password -
```

#### Using Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: <base64-encoded-url>
  openai-api-key: <base64-encoded-key>
```

### 2. Network Security

#### HTTPS/TLS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw deny 8000   # Block direct access to app
```

## Monitoring and Observability

### 1. Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/health/detailed

# Prometheus metrics
curl http://localhost:8000/metrics
```

### 2. Logging

```bash
# View application logs
docker logs agentic-startup-studio

# Follow logs in real-time
docker logs -f agentic-startup-studio

# Structured log analysis
docker logs agentic-startup-studio | jq '.level == "ERROR"'
```

### 3. Metrics Collection

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access Grafana dashboard
open http://localhost:3000

# Query Prometheus
open http://localhost:9090
```

## Performance Optimization

### 1. Resource Allocation

```yaml
# Docker Compose
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### 2. Database Optimization

```sql
-- PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

### 3. Caching Configuration

```yaml
# Redis configuration
redis:
  image: redis:7-alpine
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
  deploy:
    resources:
      limits:
        memory: 512M
```

## Backup and Recovery

### 1. Database Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump $DATABASE_URL > $BACKUP_DIR/database.sql

# Compress and upload to S3
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
aws s3 cp $BACKUP_DIR.tar.gz s3://your-backup-bucket/
```

### 2. Application State

```bash
# Backup application data
docker run --rm -v agentic_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/app_data.tar.gz /data

# Restore application data
docker run --rm -v agentic_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/app_data.tar.gz -C /
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check database connectivity
docker exec -it postgres psql -U user -d dbname -c "SELECT 1;"

# Verify connection string
python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"
```

#### 2. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits
docker update --memory=2g container_name
```

#### 3. Performance Issues
```bash
# Check application metrics
curl http://localhost:8000/metrics | grep response_time

# Analyze slow queries
docker exec -it postgres psql -U user -d dbname -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG -e DEBUG=true agentic-startup-studio

# Access debug endpoints
curl http://localhost:8000/debug/info
```

## Zero-Downtime Deployment

### 1. Blue-Green Deployment

```bash
# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Health check green environment
./scripts/health_check.sh green

# Switch traffic to green
./scripts/switch_traffic.sh green

# Remove blue environment
docker-compose -f docker-compose.blue.yml down
```

### 2. Rolling Updates (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-startup-studio
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 25%
      maxSurge: 25%
  template:
    spec:
      containers:
      - name: app
        image: agentic-startup-studio:v2.1.0
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

## Security Checklist

- [ ] Use HTTPS/TLS for all external communication
- [ ] Enable firewall and restrict ports
- [ ] Use secrets management (not environment variables in production)
- [ ] Regular security updates and patches
- [ ] Enable audit logging
- [ ] Use non-root containers
- [ ] Scan images for vulnerabilities
- [ ] Network segmentation
- [ ] Rate limiting and DDoS protection
- [ ] Regular backup testing

## Support

For deployment issues:
1. Check the logs first: `docker logs agentic-startup-studio`
2. Verify environment variables
3. Test database connectivity
4. Check resource usage
5. Review security settings

For production deployments, consider:
- Professional monitoring services
- Managed database services
- CDN for static assets
- Load balancing
- Auto-scaling policies