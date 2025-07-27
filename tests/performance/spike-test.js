import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTimeTrend = new Trend('response_time', true);

// Spike test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Normal load
    { duration: '30s', target: 100 }, // Spike to 100 users
    { duration: '30s', target: 100 }, // Stay at 100 users
    { duration: '30s', target: 200 }, // Spike to 200 users
    { duration: '30s', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 10 },   // Scale down
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s during spike
    http_req_failed: ['rate<0.2'],     // Error rate must be below 20% during spike
    errors: ['rate<0.2'],              // Custom error rate below 20%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

export default function () {
  // Focus on critical endpoints during spike
  const scenarios = [
    () => healthCheck(),
    () => healthCheck(),
    () => healthCheck(), // Weight health checks heavily
    () => listIdeas(),
    () => getMetrics(),
  ];
  
  const scenario = scenarios[Math.floor(Math.random() * scenarios.length)];
  scenario();
  
  sleep(0.5); // Reduced think time for spike test
}

function healthCheck() {
  const response = http.get(`${BASE_URL}/health`);
  
  check(response, {
    'spike health check status is 200': (r) => r.status === 200,
    'spike health check response time < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

function listIdeas() {
  const response = http.get(`${BASE_URL}/api/v1/ideas?limit=5`, {
    headers: { 'X-API-Key': API_KEY },
  });
  
  check(response, {
    'spike list ideas status is 200': (r) => r.status === 200,
    'spike list ideas response time < 2000ms': (r) => r.timings.duration < 2000,
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

function getMetrics() {
  const response = http.get(`${BASE_URL}/metrics`);
  
  check(response, {
    'spike metrics status is 200': (r) => r.status === 200,
    'spike metrics response time < 1500ms': (r) => r.timings.duration < 1500,
  }) || errorRate.add(1);
  
  responseTimeTrend.add(response.timings.duration);
}

export function setup() {
  console.log('🚀 Starting spike test...');
  return { baseUrl: BASE_URL };
}

export function teardown(data) {
  console.log('📊 Spike test completed');
}