# API Documentation v2.0 - July 2025

**Last Updated**: July 26, 2025  
**API Version**: 2.0 with Secure Gateway  
**Base URL**: `http://localhost:8001` (development) / `https://api.yourdomain.com` (production)

## Table of Contents

1. [Overview](#overview)
2. [Authentication System](#authentication-system)  
3. [API Gateway Endpoints](#api-gateway-endpoints)
4. [Business Logic APIs](#business-logic-apis)
5. [Rate Limiting and Security](#rate-limiting-and-security)
6. [Error Codes and Responses](#error-codes-and-responses)
7. [Integration Examples](#integration-examples)
8. [Migration from v1](#migration-from-v1)

---

## Overview

The Agentic Startup Studio v2.0 API provides a secure, authenticated gateway for managing startup ideas through a complete validation and development pipeline. **All APIs now require authentication** except public health endpoints.

### What's New in v2.0
- 🔐 **Mandatory Authentication**: All business APIs require JWT tokens
- ⚡ **Rate Limiting**: Per-IP and per-endpoint limits with automatic blocking
- 📊 **Enhanced Monitoring**: Prometheus metrics and distributed tracing
- 🚀 **Performance**: <50ms vector search, <200ms API response times
- 💰 **Budget Control**: Real-time cost monitoring with $62/cycle limits

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Client Apps     │    │ API Gateway     │    │ Business APIs   │
│                 │    │                 │    │                 │
│ CLI/Web/Mobile  │◄──►│ 🔐 JWT Auth     │◄──►│ Idea Management │
│ curl/Postman    │    │ ⚡ Rate Limits  │    │ Pitch Generation│
│ SDKs            │    │ 📊 Monitoring   │    │ Campaign Mgmt   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │ Prometheus      │    │ PostgreSQL      │
                    │ + Grafana       │    │ + pgvector      │
                    │ + OpenTelemetry │    │ + Redis Cache   │
                    └─────────────────┘    └─────────────────┘
```

---

## Authentication System

### Authentication Flow

All API access follows a two-step authentication process:

1. **API Key Exchange**: Trade API key for JWT token
2. **JWT Authorization**: Use token for subsequent API calls

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as API Gateway
    participant Auth as Auth Service
    
    Client->>Gateway: POST /auth/login (API Key)
    Gateway->>Auth: Validate API Key
    Auth-->>Gateway: Valid ✓
    Gateway-->>Client: JWT Token (1hr expiry)
    
    Client->>Gateway: GET /api/v1/ideas (Bearer Token)
    Gateway->>Auth: Verify JWT
    Auth-->>Gateway: Valid ✓  
    Gateway-->>Client: API Response
```

### Step 1: Obtain JWT Token

**Endpoint**: `POST /auth/login`

```bash
curl -X POST http://localhost:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key-here"}'
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Step 2: Use JWT Token

Include the JWT token in the `Authorization` header for all API calls:

```bash
curl -X GET http://localhost:8001/api/v1/ideas \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Token Management

**Verify Token**:
```bash
curl -X GET http://localhost:8001/auth/verify \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Logout/Invalidate**:
```bash
curl -X DELETE http://localhost:8001/auth/logout \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## API Gateway Endpoints

### Authentication Endpoints

#### POST /auth/login
Authenticate with API key and receive JWT token.

**Request**:
```json
{
  "api_key": "string"
}
```

**Response** (200):
```json
{
  "access_token": "string",
  "token_type": "bearer", 
  "expires_in": 3600
}
```

**Errors**:
- `401`: Invalid API key
- `429`: Rate limit exceeded

#### GET /auth/verify
Verify current JWT token validity.

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "authenticated": true,
  "session_id": "session-1627834567",
  "expires_at": 1627838167
}
```

#### DELETE /auth/logout
Invalidate current session and JWT token.

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "message": "Logged out successfully"
}
```

### System Endpoints

#### GET /health
Public health check endpoint (no authentication required).

**Response** (200):
```json
{
  "status": "healthy",
  "timestamp": "2025-07-26T12:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "vector_search": "healthy"
  }
}
```

#### GET /metrics
Prometheus metrics endpoint (authentication required).

**Headers**: `Authorization: Bearer <token>`

**Response** (200): Prometheus text format metrics

#### GET /gateway/status  
API gateway statistics (authentication required).

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "status": "healthy",
  "active_sessions": 15,
  "rate_limit_rules": 4,
  "blocked_ips": 2,
  "environment": "production"
}
```

---

## Business Logic APIs

All business APIs require authentication via JWT token in `Authorization` header.

### Idea Management

#### POST /api/v1/ideas
Submit a new startup idea for processing.

**Headers**: `Authorization: Bearer <token>`

**Request**:
```json
{
  "title": "AI-Powered Code Review Assistant",
  "description": "Automated code review using machine learning to provide instant feedback on code quality, security vulnerabilities, and best practices.",
  "category": "AI/ML",
  "problem_statement": "Developers spend 20-30% of their time on manual code reviews, leading to delayed releases and inconsistent quality standards.",
  "solution_description": "AI assistant analyzes code in real-time, providing intelligent suggestions and catching issues before human review.",
  "target_market": "Software development teams in companies with 10+ developers",
  "evidence_links": [
    "https://research.github.com/code-review-efficiency-study",
    "https://stackoverflow.com/insights/developer-survey-2024"
  ]
}
```

**Response** (201):
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "AI-Powered Code Review Assistant", 
  "description": "Automated code review using machine learning...",
  "category": "AI/ML",
  "status": "validating",
  "created_at": "2025-07-26T12:00:00Z",
  "updated_at": "2025-07-26T12:00:00Z"
}
```

**Validation Rules**:
- `title`: 10-200 characters, required
- `description`: 50-5000 characters, required  
- `category`: Valid category string
- `problem_statement`: 20-2000 characters
- `solution_description`: 20-2000 characters
- `target_market`: 10-500 characters
- `evidence_links`: Valid URLs only

#### GET /api/v1/ideas
List submitted ideas with filtering and pagination.

**Headers**: `Authorization: Bearer <token>`

**Query Parameters**:
- `limit`: Results per page (1-100, default: 50)
- `offset`: Pagination offset (default: 0)
- `category`: Filter by category
- `status_filter`: Filter by status (validated, researching, etc.)

**Response** (200):
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "AI-Powered Code Review Assistant",
    "description": "Automated code review using machine learning...",
    "category": "AI/ML", 
    "status": "researching",
    "created_at": "2025-07-26T12:00:00Z",
    "updated_at": "2025-07-26T12:05:00Z",
    "validation_score": 0.87
  }
]
```

#### GET /api/v1/ideas/{idea_id}
Get detailed information about a specific idea.

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "AI-Powered Code Review Assistant",
  "description": "Automated code review using machine learning...",
  "category": "AI/ML",
  "status": "researching", 
  "created_at": "2025-07-26T12:00:00Z",
  "updated_at": "2025-07-26T12:05:00Z",
  "validation_score": 0.87
}
```

**Errors**:
- `404`: Idea not found

### Pitch Deck Generation

#### POST /api/v1/pitch-decks
Generate a pitch deck for an existing idea.

**Headers**: `Authorization: Bearer <token>`

**Request**:
```json
{
  "idea_id": "123e4567-e89b-12d3-a456-426614174000",
  "template": "default",
  "include_financials": true,
  "include_competition": true
}
```

**Response** (200):
```json
{
  "deck_id": "deck-123e4567-1627834567",
  "idea_id": "123e4567-e89b-12d3-a456-426614174000",
  "deck_url": "/api/v1/pitch-decks/deck-123e4567-1627834567",
  "generated_at": "2025-07-26T12:10:00Z",
  "status": "generating"
}
```

#### GET /api/v1/pitch-decks/{deck_id}
Get pitch deck status and download link.

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "deck_id": "deck-123e4567-1627834567",
  "status": "completed",
  "download_url": "/api/v1/pitch-decks/deck-123e4567-1627834567/download",
  "generated_at": "2025-07-26T12:10:00Z",
  "format": "pdf"
}
```

### Campaign Management

#### POST /api/v1/campaigns
Create a marketing campaign for an idea.

**Headers**: `Authorization: Bearer <token>`

**Request**:
```json
{
  "idea_id": "123e4567-e89b-12d3-a456-426614174000",
  "campaign_type": "google_ads",
  "budget": 45.00,
  "duration_days": 30,
  "target_audience": "Software developers, CTOs, development team leads"
}
```

**Response** (200):
```json
{
  "campaign_id": "campaign-123e4567-1627834567",
  "idea_id": "123e4567-e89b-12d3-a456-426614174000", 
  "campaign_type": "google_ads",
  "status": "creating",
  "budget": 45.00,
  "created_at": "2025-07-26T12:15:00Z"
}
```

#### GET /api/v1/campaigns/{campaign_id}
Get campaign performance metrics and status.

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "campaign_id": "campaign-123e4567-1627834567",
  "status": "active",
  "metrics": {
    "impressions": 10000,
    "clicks": 250, 
    "conversions": 15,
    "spend": 45.67,
    "ctr": 2.5,
    "conversion_rate": 6.0
  },
  "updated_at": "2025-07-26T12:20:00Z"
}
```

---

## Rate Limiting and Security

### Rate Limits by Endpoint

| Endpoint | Requests/Minute | Requests/Hour | Burst Limit |
|----------|----------------|---------------|-------------|
| `/health` | 120 | 2000 | 20 |
| `/metrics` | 30 | 500 | 10 |
| `/api/v1/ideas` | 30 | 200 | 10 |
| `/api/v1/pitch` | 5 | 50 | 3 |
| **Default** | 60 | 1000 | 10 |

### Rate Limit Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 29
X-RateLimit-Reset: 1627834627
```

### IP Blocking

- **Violation Threshold**: 5+ rate limit violations in 5 minutes
- **Block Duration**: 5 minutes automatic release
- **Override**: Contact admin for manual unblock

### Security Features

**Request Security**:
- HTTPS required in production
- CORS protection with configurable origins
- Input sanitization and validation
- SQL injection prevention

**Authentication Security**:
- JWT tokens expire after 1 hour
- Session tracking with automatic cleanup  
- API key rotation support
- Audit logging for all authentication events

**Monitoring Security**:
- Real-time threat detection
- Prometheus security metrics
- Comprehensive audit trails
- Anomaly detection alerts

---

## Error Codes and Responses

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Valid auth but insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "ValidationError",
  "message": "Title must be between 10 and 200 characters",
  "field": "title",
  "code": "VALIDATION_FAILED",
  "timestamp": "2025-07-26T12:00:00Z",
  "correlation_id": "req-1627834567"
}
```

### Common Errors

**Authentication Errors**:
```json
{
  "error": "AuthenticationError",
  "message": "Invalid API key provided",
  "code": "INVALID_API_KEY",
  "timestamp": "2025-07-26T12:00:00Z"
}
```

**Rate Limit Errors**:
```json
{
  "error": "RateLimitError", 
  "message": "Rate limit exceeded. Max 30 requests/minute",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60,
  "timestamp": "2025-07-26T12:00:00Z"
}
```

**Validation Errors**:
```json
{
  "error": "ValidationError",
  "message": "Invalid input data",
  "code": "VALIDATION_FAILED",
  "details": [
    {
      "field": "title",
      "message": "Title must be between 10 and 200 characters",
      "provided_value": "AI"
    }
  ],
  "timestamp": "2025-07-26T12:00:00Z"
}
```

---

## Integration Examples

### Python SDK Example

```python
import requests
import json

class StartupStudioAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.token = None
        self.authenticate()
    
    def authenticate(self):
        """Get JWT token using API key."""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"api_key": self.api_key}
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
    
    def _headers(self):
        """Get authenticated headers."""
        return {"Authorization": f"Bearer {self.token}"}
    
    def create_idea(self, idea_data):
        """Submit a new startup idea."""
        response = requests.post(
            f"{self.base_url}/api/v1/ideas",
            json=idea_data,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def list_ideas(self, limit=50, category=None):
        """List ideas with optional filtering."""
        params = {"limit": limit}
        if category:
            params["category"] = category
            
        response = requests.get(
            f"{self.base_url}/api/v1/ideas",
            params=params,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

# Usage
api = StartupStudioAPI("http://localhost:8001", "your-api-key")

# Create new idea
idea = api.create_idea({
    "title": "AI-Powered Fitness Tracker",
    "description": "Smart fitness tracking with personalized AI coaching...",
    "category": "HealthTech",
    "problem_statement": "Generic fitness apps don't provide personalized guidance",
    "solution_description": "AI analyzes user data to provide custom workout plans",
    "target_market": "Health-conscious professionals aged 25-45"
})

print(f"Created idea: {idea['id']}")

# List all ideas
ideas = api.list_ideas(limit=10, category="HealthTech")
print(f"Found {len(ideas)} HealthTech ideas")
```

### cURL Examples

**Complete Workflow**:

```bash
#!/bin/bash

BASE_URL="http://localhost:8001"
API_KEY="your-api-key-here"

# 1. Authenticate and get token
TOKEN=$(curl -s -X POST $BASE_URL/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"api_key\": \"$API_KEY\"}" | \
  jq -r '.access_token')

echo "Got token: ${TOKEN:0:20}..."

# 2. Create new idea
IDEA_ID=$(curl -s -X POST $BASE_URL/api/v1/ideas \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Smart Home Energy Optimizer",
    "description": "AI-powered energy management system that learns household patterns to optimize electricity usage and reduce costs.",
    "category": "CleanTech",
    "problem_statement": "Homeowners waste 20-30% of energy due to inefficient usage patterns",
    "solution_description": "ML algorithms predict and optimize energy consumption in real-time",
    "target_market": "Homeowners with smart home devices and high energy bills"
  }' | jq -r '.id')

echo "Created idea: $IDEA_ID"

# 3. Get idea details
curl -s -X GET $BASE_URL/api/v1/ideas/$IDEA_ID \
  -H "Authorization: Bearer $TOKEN" | jq '.'

# 4. Generate pitch deck
DECK_ID=$(curl -s -X POST $BASE_URL/api/v1/pitch-decks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"idea_id\": \"$IDEA_ID\",
    \"template\": \"default\",
    \"include_financials\": true,
    \"include_competition\": true
  }" | jq -r '.deck_id')

echo "Started pitch deck generation: $DECK_ID"

# 5. Check deck status
curl -s -X GET $BASE_URL/api/v1/pitch-decks/$DECK_ID \
  -H "Authorization: Bearer $TOKEN" | jq '.'
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

class StartupStudioAPI {
  constructor(baseURL, apiKey) {
    this.baseURL = baseURL;
    this.apiKey = apiKey;
    this.token = null;
  }

  async authenticate() {
    const response = await axios.post(`${this.baseURL}/auth/login`, {
      api_key: this.apiKey
    });
    this.token = response.data.access_token;
  }

  get headers() {
    return { Authorization: `Bearer ${this.token}` };
  }

  async createIdea(ideaData) {
    const response = await axios.post(
      `${this.baseURL}/api/v1/ideas`,
      ideaData,
      { headers: this.headers }
    );
    return response.data;
  }

  async listIdeas(options = {}) {
    const response = await axios.get(
      `${this.baseURL}/api/v1/ideas`,
      { 
        params: options,
        headers: this.headers 
      }
    );
    return response.data;
  }

  async generatePitchDeck(ideaId, options = {}) {
    const response = await axios.post(
      `${this.baseURL}/api/v1/pitch-decks`,
      { idea_id: ideaId, ...options },
      { headers: this.headers }
    );
    return response.data;
  }
}

// Usage
async function main() {
  const api = new StartupStudioAPI('http://localhost:8001', 'your-api-key');
  
  try {
    await api.authenticate();
    
    const idea = await api.createIdea({
      title: 'Voice-Controlled Smart Mirror',
      description: 'Interactive mirror with voice commands, health tracking, and personalized information display.',
      category: 'Consumer',
      problem_statement: 'Morning routines are inefficient and lack personalization',
      solution_description: 'Smart mirror provides personalized info and health insights',
      target_market: 'Tech-savvy consumers interested in smart home automation'
    });
    
    console.log('Created idea:', idea.id);
    
    const deck = await api.generatePitchDeck(idea.id, {
      template: 'default',
      include_financials: true
    });
    
    console.log('Pitch deck generation started:', deck.deck_id);
    
  } catch (error) {
    if (error.response?.status === 429) {
      console.error('Rate limited. Retry after:', error.response.headers['retry-after']);
    } else {
      console.error('API error:', error.response?.data || error.message);
    }
  }
}

main();
```

---

## Migration from v1

### Breaking Changes in v2.0

1. **Authentication Required**: All business APIs now require JWT tokens
2. **Rate Limiting**: New rate limits may block high-volume usage  
3. **New Error Format**: Standardized error response structure
4. **Endpoint Changes**: Some endpoints moved under `/api/v1/` prefix

### Migration Steps

**Step 1: Update Authentication**
```diff
# Old (v1)
- curl -X GET http://localhost:8000/ideas

# New (v2)  
+ # First get token
+ TOKEN=$(curl -X POST http://localhost:8001/auth/login \
+   -d '{"api_key": "your-key"}' | jq -r '.access_token')
+ 
+ # Then use token
+ curl -X GET http://localhost:8001/api/v1/ideas \
+   -H "Authorization: Bearer $TOKEN"
```

**Step 2: Update Error Handling**
```diff
# Old error format
- {"error": "Validation failed"}

# New error format
+ {
+   "error": "ValidationError",
+   "message": "Title must be between 10 and 200 characters", 
+   "code": "VALIDATION_FAILED",
+   "field": "title"
+ }
```

**Step 3: Handle Rate Limits**
```python
import time
import requests

def api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

### Compatibility Layer

For gradual migration, v1 endpoints are still available but deprecated:

```bash
# Deprecated v1 health endpoint (will be removed)
curl -X GET http://localhost:8001/health

# New v2 health endpoint (recommended)  
curl -X GET http://localhost:8001/health
```

**Migration Timeline**:
- **Phase 1** (Current): v2 available, v1 deprecated with warnings
- **Phase 2** (Aug 2025): v1 endpoints return 301 redirects
- **Phase 3** (Sep 2025): v1 endpoints removed entirely

---

**Last Updated**: July 26, 2025  
**Support**: For API questions, create an issue in the repository or contact the development team.