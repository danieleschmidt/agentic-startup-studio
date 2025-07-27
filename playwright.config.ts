import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for Agentic Startup Studio
 * E2E Testing Configuration with multiple browsers and environments
 */
export default defineConfig({
  testDir: './tests/e2e',
  
  // Run tests in files in parallel
  fullyParallel: true,
  
  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,
  
  // Retry on CI only
  retries: process.env.CI ? 2 : 0,
  
  // Number of parallel workers - optimize for CI vs local
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['junit', { outputFile: 'test-results/e2e-results.xml' }],
    process.env.CI ? ['github'] : ['list']
  ],
  
  // Global test timeout
  timeout: 30 * 1000,
  
  // Global expect timeout
  expect: {
    timeout: 5000,
  },
  
  // Shared settings for all the projects below
  use: {
    // Base URL for all tests
    baseURL: process.env.BASE_URL || 'http://localhost:8000',
    
    // Collect trace when retrying the failed test
    trace: 'on-first-retry',
    
    // Take screenshot on failure
    screenshot: 'only-on-failure',
    
    // Record video on failure
    video: 'retain-on-failure',
    
    // API request context
    extraHTTPHeaders: {
      'Accept': 'application/json',
      'User-Agent': 'Playwright E2E Tests'
    },
    
    // Ignore HTTPS errors for local development
    ignoreHTTPSErrors: true,
  },

  // Configure projects for major browsers
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    
    // Mobile testing
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
    
    // API testing
    {
      name: 'api',
      testDir: './tests/e2e/api',
      use: {
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000/api/v1',
      },
    },
  ],

  // Global setup and teardown
  globalSetup: require.resolve('./tests/e2e/global-setup.ts'),
  globalTeardown: require.resolve('./tests/e2e/global-teardown.ts'),

  // Development server configuration
  webServer: {
    command: 'python scripts/serve_api.py --port 8000',
    port: 8000,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000, // 2 minutes timeout
    env: {
      ENVIRONMENT: 'test',
      LOG_LEVEL: 'WARNING',
    },
  },
  
  // Output directories
  outputDir: 'test-results/e2e-output/',
  
  // Test artifacts
  preserveOutput: 'failures-only',
});