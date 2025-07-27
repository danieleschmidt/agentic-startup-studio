import { FullConfig } from '@playwright/test';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

/**
 * Global teardown for Playwright E2E tests
 * Runs after all tests to clean up the environment
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global test cleanup...');
  
  try {
    // Clean up test data
    console.log('🗑️ Cleaning up test data...');
    await execAsync('python -c "from pipeline.storage.idea_repository import IdeaRepository; repo = IdeaRepository(); repo.cleanup_test_data()"');
    
    // Reset test database
    console.log('🔄 Resetting test database...');
    await execAsync('python -c "from pipeline.config.settings import cleanup_test_database; cleanup_test_database()"');
    
    // Generate test report
    console.log('📊 Generating test report...');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    await execAsync(`echo '{"timestamp": "${timestamp}", "cleanup": "completed"}' > test-results/teardown-${timestamp}.json`);
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
    console.error('Continuing despite teardown errors...');
  }
}

export default globalTeardown;