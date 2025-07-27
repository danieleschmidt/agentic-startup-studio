import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTimeTrend = new Trend('response_time', true);

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 }, // Ramp up to 10 users
    { duration: '5m', target: 10 }, // Stay at 10 users
    { duration: '2m', target: 20 }, // Ramp up to 20 users
    { duration: '5m', target: 20 }, // Stay at 20 users
    { duration: '2m', target: 0 },  // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.1'],    // Error rate must be below 10%
    errors: ['rate<0.1'],             // Custom error rate below 10%
  },
};

// Test data
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test scenarios
export default function () {
  const testScenarios = [
    healthCheck,
    createIdea,
    listIdeas,
    getIdea,
    searchIdeas,
  ];
  
  // Run random scenario
  const scenario = testScenarios[Math.floor(Math.random() * testScenarios.length)];
  scenario();
  
  sleep(1); // Think time between requests
}

/**
 * Health Check Load Test
 */
function healthCheck() {
  const response = http.get(`${BASE_URL}/health`);
  
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 200ms': (r) => r.timings.duration < 200,
    'health check returns healthy status': (r) => JSON.parse(r.body).status === 'healthy',
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

/**
 * Create Idea Load Test
 */
function createIdea() {
  const payload = JSON.stringify({
    title: `Load Test Idea ${Date.now()}`,
    description: 'A test idea created during load testing',
    category: 'ai_ml',
    problem: 'Testing system performance under load',
    solution: 'Automated load testing with k6',
    market: 'Performance testing market',
    evidence: ['https://example.com/evidence1']
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
  };
  
  const response = http.post(`${BASE_URL}/api/v1/ideas`, payload, params);
  
  check(response, {
    'create idea status is 201': (r) => r.status === 201,
    'create idea response time < 2000ms': (r) => r.timings.duration < 2000,
    'create idea returns idea object': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('id') && data.hasOwnProperty('title');
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

/**
 * List Ideas Load Test
 */
function listIdeas() {
  const params = {
    headers: {
      'X-API-Key': API_KEY,
    },
  };
  
  const response = http.get(`${BASE_URL}/api/v1/ideas?limit=10`, params);
  
  check(response, {
    'list ideas status is 200': (r) => r.status === 200,
    'list ideas response time < 500ms': (r) => r.timings.duration < 500,
    'list ideas returns array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.ideas);
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

/**
 * Get Single Idea Load Test
 */
function getIdea() {
  // First, get a list of ideas to pick a random one
  const listResponse = http.get(`${BASE_URL}/api/v1/ideas?limit=5`, {
    headers: { 'X-API-Key': API_KEY },
  });
  
  if (listResponse.status === 200) {
    try {
      const data = JSON.parse(listResponse.body);
      if (data.ideas && data.ideas.length > 0) {
        const randomIdea = data.ideas[Math.floor(Math.random() * data.ideas.length)];
        
        const response = http.get(`${BASE_URL}/api/v1/ideas/${randomIdea.id}`, {
          headers: { 'X-API-Key': API_KEY },
        });
        
        check(response, {
          'get idea status is 200': (r) => r.status === 200,
          'get idea response time < 300ms': (r) => r.timings.duration < 300,
          'get idea returns idea object': (r) => {
            try {
              const ideaData = JSON.parse(r.body);
              return ideaData.hasOwnProperty('id') && ideaData.id === randomIdea.id;
            } catch (e) {
              return false;
            }
          },
        }) || errorRate.add(1);
        
        responseTimeTrend.add(response.timings.duration);
      }
    } catch (e) {
      errorRate.add(1);
    }
  }
}

/**
 * Search Ideas Load Test
 */
function searchIdeas() {
  const searchTerms = ['AI', 'machine learning', 'SaaS', 'fintech', 'healthcare'];
  const randomTerm = searchTerms[Math.floor(Math.random() * searchTerms.length)];
  
  const params = {
    headers: {
      'X-API-Key': API_KEY,
    },
  };
  
  const response = http.get(`${BASE_URL}/api/v1/ideas?search=${randomTerm}&limit=10`, params);
  
  check(response, {
    'search ideas status is 200': (r) => r.status === 200,
    'search ideas response time < 1000ms': (r) => r.timings.duration < 1000,
    'search ideas returns results': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('ideas') && Array.isArray(data.ideas);
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

/**
 * Setup function - runs once before the test
 */
export function setup() {
  console.log('🚀 Starting performance test setup...');
  
  // Verify API is accessible
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`API health check failed: ${healthResponse.status}`);
  }
  
  console.log('✅ Performance test setup completed');
  return { baseUrl: BASE_URL };
}

/**
 * Teardown function - runs once after the test
 */
export function teardown(data) {
  console.log('🧹 Starting performance test teardown...');
  console.log('📊 Test completed successfully');
}