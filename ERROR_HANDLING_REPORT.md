# Error Handling Improvement Report

## Executive Summary

This report identifies areas in the Python codebase that need better error handling, prioritized by risk level. The analysis focused on:
- External API calls without proper exception handling
- Database operations without connection failure handling
- File operations without try/catch blocks
- Network operations without timeout or retry logic
- Missing validation in public API endpoints
- Configuration loading without error handling

## Critical Issues (Priority 1 - Immediate Action Required)

### 1. Database Operations Without Proper Error Handling

#### `/root/repo/core/idea_ledger.py`

**Issue**: Database operations lack error handling for connection failures and transaction rollbacks.

**Lines**: 46-50, 59-61, 69-72, 82-95, 104-110

**Required Error Handling**:
- Add connection failure handling
- Implement transaction rollback on errors
- Add retry logic for transient database errors

**Example Fix**:
```python
def add_idea(idea_create: IdeaCreate) -> Idea:
    """Adds a new Idea to the database with proper error handling."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            db_idea = Idea.model_validate(idea_create)
            with Session(engine) as session:
                session.add(db_idea)
                session.commit()
                session.refresh(db_idea)
            return db_idea
        except SQLAlchemyError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            logger.error(f"Database error after {max_retries} attempts: {e}")
            raise DatabaseConnectionError(f"Failed to add idea: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error adding idea: {e}")
            raise
```

### 2. External API Calls Without Exception Handling

#### `/root/repo/tools/dittofeed_api.py`

**Issue**: Direct API call without error handling or timeout.

**Line**: 12

**Required Error Handling**:
- Add request timeout
- Handle connection errors
- Add retry logic
- Validate response status

**Example Fix**:
```python
def create_journey(payload):
    try:
        r = requests.post(
            f"{BASE}/api/v1/journeys",
            headers={"Authorization": f"Bearer {TOK}"},
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise APITimeoutError("Dittofeed API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise APIError(f"Failed to create journey: {str(e)}")
```

## High Priority Issues (Priority 2 - Address Within Sprint)

### 3. Missing Validation in Public API Endpoints

#### `/root/repo/pipeline/api/health_server.py`

**Issue**: No input validation or error handling in API endpoints.

**Lines**: 18-26, 29-35

**Required Error Handling**:
- Add exception handling around infrastructure calls
- Implement proper HTTP error responses
- Add request validation

**Example Fix**:
```python
@app.get("/health")
async def health() -> dict:
    """Return overall system health with proper error handling."""
    try:
        status = await get_infrastructure_health()
        gauge_value = 1 if status.get("status") == "healthy" else 0
        if status.get("status") == "unhealthy":
            gauge_value = -1
        health_gauge.set(gauge_value)
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": "Health check failed",
            "timestamp": datetime.utcnow().isoformat()
        }
```

### 4. Network Operations Without Proper Security Validation

#### `/root/repo/pipeline/services/evidence_collector.py`

**Issue**: While SSRF protection exists, some network operations lack comprehensive error handling.

**Lines**: 575-598 (partial error handling exists but could be improved)

**Required Improvements**:
- Add more specific exception handling for different failure types
- Implement circuit breaker pattern for repeated failures
- Add metrics for network failures

### 5. File Operations Without Try/Catch Blocks

#### `/root/repo/pipeline/services/campaign_generator_async.py`

**Issue**: File read operation without comprehensive error handling.

**Lines**: 246-247

**Required Error Handling**:
- Handle file not found
- Handle permission errors
- Handle encoding errors

**Example Fix**:
```python
try:
    async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
        content = await f.read()
except FileNotFoundError:
    logger.error(f"Template file not found: {template_path}")
    raise TemplateNotFoundError(f"Template not found: {template_path}")
except PermissionError:
    logger.error(f"Permission denied accessing template: {template_path}")
    raise TemplateAccessError(f"Cannot access template: {template_path}")
except Exception as e:
    logger.error(f"Error reading template file: {e}")
    raise TemplateError(f"Failed to read template: {str(e)}")
```

## Medium Priority Issues (Priority 3 - Technical Debt)

### 6. Configuration Loading Without Error Handling

#### `/root/repo/pipeline/config/settings.py`

**Issue**: Settings loading should have fallback mechanisms.

**Recommendation**: 
- Add validation for required settings
- Implement graceful fallbacks for optional settings
- Add early validation at startup

### 7. Async Operations Without Proper Cleanup

#### `/root/repo/pipeline/main_pipeline_async.py`

**Issue**: Connection pool initialization lacks cleanup on failure.

**Lines**: 211-224

**Required Error Handling**:
- Ensure connection pool is properly closed on errors
- Add connection health checks
- Implement connection pool recovery

## Low Priority Issues (Priority 4 - Best Practices)

### 8. Missing Retry Logic in Service Calls

Several service-to-service calls lack retry logic:
- Campaign generation service calls
- Evidence collection service calls
- Budget sentinel tracking calls

**Recommendation**: Implement a standardized retry decorator for all service calls.

## Recommendations

### 1. Implement Standard Error Handling Patterns

Create base exception classes:
```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DatabaseError(PipelineError):
    """Database-related errors."""
    pass

class APIError(PipelineError):
    """External API errors."""
    pass

class ValidationError(PipelineError):
    """Input validation errors."""
    pass
```

### 2. Add Circuit Breaker Pattern

Implement circuit breakers for all external service calls to prevent cascading failures.

### 3. Standardize Logging

Ensure all error handlers log with appropriate context:
```python
logger.error(
    "Operation failed",
    extra={
        "operation": "add_idea",
        "error_type": type(e).__name__,
        "error_message": str(e),
        "idea_id": idea_id,
        "retry_count": attempt
    }
)
```

### 4. Add Health Checks

Implement health checks for all external dependencies:
- Database connectivity
- External API availability
- File system access
- Network connectivity

### 5. Implement Graceful Degradation

Services should continue operating with reduced functionality when dependencies fail.

## Implementation Priority

1. **Week 1**: Fix all Priority 1 issues (database and API error handling)
2. **Week 2**: Address Priority 2 issues (API validation and network operations)
3. **Week 3**: Implement standard error handling patterns and circuit breakers
4. **Week 4**: Add comprehensive health checks and monitoring

## Testing Requirements

For each error handling implementation:
1. Unit tests for error scenarios
2. Integration tests for dependency failures
3. Load tests to verify retry logic
4. Chaos engineering tests for resilience

## Metrics to Track

- Error rate by service
- Retry success rate
- Circuit breaker state changes
- Mean time to recovery (MTTR)
- Dependency failure impact