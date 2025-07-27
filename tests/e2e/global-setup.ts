import { FullConfig } from '@playwright/test';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

/**
 * Global setup for Playwright E2E tests
 * Runs before all tests to prepare the environment
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 Setting up global test environment...');
  
  try {
    // Set environment variables for testing
    process.env.ENVIRONMENT = 'test';
    process.env.LOG_LEVEL = 'WARNING';
    process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || 'postgresql://studio:studio@localhost:5432/studio_test';
    
    // Initialize test database
    console.log('📊 Initializing test database...');
    await execAsync('python -c "from pipeline.config.settings import setup_test_database; setup_test_database()"');
    
    // Run database migrations
    console.log('🔄 Running database migrations...');
    await execAsync('python scripts/setup_production_secrets.py --test-mode');
    
    // Seed test data
    console.log('🌱 Seeding test data...');
    await execAsync('python scripts/seed_idea.py "E2E Test Idea" --test-mode');
    
    // Health check
    console.log('🔍 Running health checks...');
    const healthResult = await execAsync('python scripts/run_health_checks.py --format json');
    const healthData = JSON.parse(healthResult.stdout);
    
    if (!healthData.overall_health || healthData.overall_health !== 'healthy') {
      throw new Error(`Health check failed: ${JSON.stringify(healthData)}`);
    }
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  }
}

export default globalSetup;