# API Gateway Deployment Guide

## Overview

The API Gateway provides centralized authentication, rate limiting, and request routing for all Agentic Startup Studio APIs.

## Features Implemented

✅ **Authentication Middleware**
- API key-based authentication
- JWT token generation and validation
- Session management with automatic cleanup

✅ **Rate Limiting**
- Per-IP rate limiting
- Per-endpoint rate limits
- Burst protection
- Automatic IP blocking for violations

✅ **Request/Response Logging**
- Comprehensive request logging with correlation IDs
- Prometheus metrics collection
- Performance monitoring

✅ **Security Controls**
- CORS middleware configuration
- Trusted host validation for production
- Request sanitization and validation

## API Endpoints

### Authentication
- `POST /auth/login` - Authenticate with API key, receive JWT token
- `GET /auth/verify` - Verify current JWT token
- `DELETE /auth/logout` - Invalidate current session

### Health & Monitoring
- `GET /health` - Public health check (unauthenticated)
- `GET /metrics` - Prometheus metrics (authenticated)
- `GET /gateway/status` - Gateway status and statistics (authenticated)

### Business APIs
- `POST /api/v1/ideas` - Submit new startup idea (authenticated)
- `GET /api/v1/ideas` - List ideas with filtering (authenticated) 
- `GET /api/v1/ideas/{id}` - Get specific idea (authenticated)
- `POST /api/v1/pitch-decks` - Generate pitch deck (authenticated)
- `POST /api/v1/campaigns` - Create marketing campaign (authenticated)

## Deployment

### Environment Variables

Required for production:
```bash
ENVIRONMENT=production
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
API_KEYS=key1,key2,key3
ALLOWED_ORIGINS=yourdomain.com,api.yourdomain.com
```

Optional configuration:
```bash
GATEWAY_PORT=8001
HOST_INTERFACE=0.0.0.0
DB_HOST=localhost
DB_PASSWORD=your-db-password
```

### Docker Deployment

Add to docker-compose.yml:
```yaml
services:
  api-gateway:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - API_KEYS=${API_KEYS}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
    command: python -m uvicorn pipeline.api.gateway:app --host 0.0.0.0 --port 8001
    depends_on:
      - postgres
      - prometheus
```

### Start Gateway

Development:
```bash
python pipeline/api/gateway.py
```

Production:
```bash
uvicorn pipeline.api.gateway:app --host 0.0.0.0 --port 8001
```

## Rate Limits

Default rate limits per endpoint:
- `/health`: 120 req/min, 2000 req/hour
- `/metrics`: 30 req/min, 500 req/hour  
- `/api/v1/ideas`: 30 req/min, 200 req/hour
- `/api/v1/pitch`: 5 req/min, 50 req/hour
- Others: 60 req/min, 1000 req/hour

## Security

- All API endpoints except `/health` require authentication
- JWT tokens expire after 1 hour
- Rate limiting prevents abuse with automatic IP blocking
- CORS configured for allowed origins only
- Request/response logging for security monitoring

## Monitoring

Prometheus metrics exported:
- `gateway_requests_total` - Total requests by method/endpoint/status
- `gateway_request_duration_seconds` - Request duration histogram
- `gateway_rate_limit_hits_total` - Rate limit violations
- `gateway_auth_failures_total` - Authentication failures
- `gateway_active_sessions` - Active authenticated sessions

## Testing

Run test suite:
```bash
python -m pytest tests/api/test_gateway.py -v
```

## Rollback Plan

If issues occur with the gateway:

1. **Immediate rollback**: Use legacy health server
   ```bash
   python pipeline/api/health_server.py
   ```

2. **Disable authentication**: Set `API_KEYS=""` environment variable

3. **Disable rate limiting**: Restart with `ENVIRONMENT=development`

## Security Audit

The API Gateway implementation follows security best practices:

- ✅ Authentication required for sensitive endpoints
- ✅ JWT tokens with expiration
- ✅ Rate limiting with burst protection  
- ✅ CORS properly configured
- ✅ No secrets in logs or responses
- ✅ Input validation on all endpoints
- ✅ Comprehensive security monitoring

Completed: **AUTH-001 - API Gateway with Authentication and Rate Limiting**