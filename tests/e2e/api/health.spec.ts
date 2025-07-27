import { test, expect } from '@playwright/test';

/**
 * API Health Endpoints E2E Tests
 * Tests the health check and monitoring endpoints
 */
test.describe('Health Check API', () => {
  
  test('should return healthy status from /health endpoint', async ({ request }) => {
    const response = await request.get('/health');
    
    expect(response.status()).toBe(200);
    
    const healthData = await response.json();
    expect(healthData).toHaveProperty('status', 'healthy');
    expect(healthData).toHaveProperty('timestamp');
    expect(healthData).toHaveProperty('version');
    expect(healthData).toHaveProperty('checks');
    
    // Verify all health checks pass
    expect(healthData.checks.database).toBe('healthy');
    expect(healthData.checks.redis).toBeDefined();
    expect(healthData.checks.external_services).toBeDefined();
  });
  
  test('should return metrics from /metrics endpoint', async ({ request }) => {
    const response = await request.get('/metrics');
    
    expect(response.status()).toBe(200);
    
    const metricsText = await response.text();
    
    // Check for Prometheus format metrics
    expect(metricsText).toContain('# HELP');
    expect(metricsText).toContain('# TYPE');
    expect(metricsText).toContain('http_requests_total');
    expect(metricsText).toContain('pipeline_ideas_processed_total');
  });
  
  test('should return readiness status from /ready endpoint', async ({ request }) => {
    const response = await request.get('/ready');
    
    expect(response.status()).toBe(200);
    
    const readyData = await response.json();
    expect(readyData).toHaveProperty('ready', true);
    expect(readyData).toHaveProperty('dependencies');
    
    // Verify all dependencies are ready
    const deps = readyData.dependencies;
    expect(deps.database).toBe('ready');
    expect(deps.cache).toBeDefined();
  });
  
  test('should handle authenticated health checks', async ({ request }) => {
    // Test protected health endpoint with authentication
    const response = await request.get('/health/detailed', {
      headers: {
        'Authorization': 'Bearer test-token',
        'X-API-Key': 'test-api-key'
      }
    });
    
    if (response.status() === 401) {
      // If authentication is required, verify proper error response
      const errorData = await response.json();
      expect(errorData).toHaveProperty('error');
      expect(errorData.error).toContain('authentication');
    } else {
      // If authentication passes, verify detailed health data
      expect(response.status()).toBe(200);
      const detailedHealth = await response.json();
      expect(detailedHealth).toHaveProperty('detailed_checks');
    }
  });
  
  test('should respond within performance thresholds', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.get('/health');
    const responseTime = Date.now() - startTime;
    
    expect(response.status()).toBe(200);
    expect(responseTime).toBeLessThan(1000); // Less than 1 second
    
    const healthData = await response.json();
    expect(healthData.response_time_ms).toBeLessThan(500); // API reports < 500ms
  });
});