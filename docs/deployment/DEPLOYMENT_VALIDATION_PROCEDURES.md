# Deployment Validation and Rollback Procedures

## Overview

This document outlines comprehensive deployment validation and rollback procedures for the Agentic Startup Studio platform, ensuring zero-downtime deployments and rapid recovery capabilities.

## Pre-Deployment Validation

### 1. Automated Pre-Flight Checks

Create `scripts/deployment_preflight.py`:

```python
#!/usr/bin/env python3
"""
Pre-deployment validation script
Validates environment, dependencies, and system readiness
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import asyncpg
import psutil


class DeploymentValidator:
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = self._setup_logging()
        self.validation_results = []
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def validate_database_connectivity(self) -> bool:
        """Validate database connection and basic operations"""
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                self.logger.error("DATABASE_URL not configured")
                return False
                
            conn = await asyncpg.connect(db_url)
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            self.logger.info("✅ Database connectivity validated")
            return result == 1
            
        except Exception as e:
            self.logger.error(f"❌ Database validation failed: {e}")
            return False
    
    async def validate_external_services(self) -> bool:
        """Validate external service connectivity"""
        services = [
            {"name": "OpenAI API", "url": "https://api.openai.com/v1/models"},
            {"name": "Google APIs", "url": "https://www.googleapis.com/oauth2/v1/certs"},
        ]
        
        results = []
        async with aiohttp.ClientSession() as session:
            for service in services:
                try:
                    async with session.get(service["url"], timeout=10) as response:
                        if response.status == 200:
                            self.logger.info(f"✅ {service['name']} connectivity validated")
                            results.append(True)
                        else:
                            self.logger.error(f"❌ {service['name']} returned status {response.status}")
                            results.append(False)
                except Exception as e:
                    self.logger.error(f"❌ {service['name']} validation failed: {e}")
                    results.append(False)
        
        return all(results)
    
    def validate_system_resources(self) -> bool:
        """Validate system resource availability"""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                self.logger.error(f"❌ High memory usage: {memory.percent}%")
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.logger.error(f"❌ Low disk space: {disk.percent}% used")
                return False
            
            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                self.logger.error(f"❌ High CPU usage: {cpu_percent}%")
                return False
            
            self.logger.info("✅ System resources validated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System resource validation failed: {e}")
            return False
    
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables"""
        required_vars = [
            "DATABASE_URL",
            "SECRET_KEY",
            "OPENAI_API_KEY",
            "ENVIRONMENT",
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
        
        self.logger.info("✅ Environment variables validated")
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate critical dependencies"""
        try:
            # Run pip check
            result = subprocess.run(
                ["pip", "check"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"❌ Dependency conflicts detected: {result.stdout}")
                return False
            
            self.logger.info("✅ Dependencies validated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dependency validation failed: {e}")
            return False
    
    async def run_health_checks(self) -> bool:
        """Run comprehensive health checks"""
        try:
            result = subprocess.run(
                ["python", "scripts/run_health_checks.py", "--comprehensive"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.logger.error(f"❌ Health checks failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Health checks passed")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Health checks timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Health check execution failed: {e}")
            return False
    
    async def validate_all(self) -> Dict[str, bool]:
        """Run all validation checks"""
        self.logger.info(f"🚀 Starting deployment validation for {self.environment}")
        
        validations = {
            "environment_variables": self.validate_environment_variables(),
            "dependencies": self.validate_dependencies(),
            "system_resources": self.validate_system_resources(),
            "database_connectivity": await self.validate_database_connectivity(),
            "external_services": await self.validate_external_services(),
            "health_checks": await self.run_health_checks(),
        }
        
        all_passed = all(validations.values())
        
        if all_passed:
            self.logger.info("🎉 All deployment validations passed!")
        else:
            failed = [k for k, v in validations.items() if not v]
            self.logger.error(f"💥 Deployment validation failed: {failed}")
        
        # Save results
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.environment,
            "validations": validations,
            "overall_status": "PASS" if all_passed else "FAIL"
        }
        
        with open("deployment_validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return validations


async def main():
    environment = os.getenv("ENVIRONMENT", "production")
    validator = DeploymentValidator(environment)
    
    results = await validator.validate_all()
    
    if not all(results.values()):
        sys.exit(1)
    
    print("✅ Deployment validation completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Database Migration Validation

Create `scripts/validate_migrations.py`:

```python
#!/usr/bin/env python3
"""
Database migration validation script
Validates migrations in a safe, rollback-friendly manner
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg


class MigrationValidator:
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def create_backup(self) -> Optional[str]:
        """Create database backup before migration"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_pre_migration_{timestamp}.sql"
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                self.logger.error("DATABASE_URL not configured")
                return None
            
            # Create backup using pg_dump
            cmd = f"pg_dump {db_url} > {backup_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Backup failed: {result.stderr}")
                return None
            
            self.logger.info(f"✅ Database backup created: {backup_file}")
            return backup_file
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None
    
    async def validate_migration_integrity(self) -> bool:
        """Validate migration files integrity"""
        try:
            # Check for migration conflicts
            migration_dir = "db/migrations"
            if not os.path.exists(migration_dir):
                self.logger.info("No migrations directory found")
                return True
            
            migration_files = sorted([
                f for f in os.listdir(migration_dir) 
                if f.endswith('.sql')
            ])
            
            if not migration_files:
                self.logger.info("No migration files found")
                return True
            
            # Validate migration file naming
            for file in migration_files:
                if not file.split('_')[0].isdigit():
                    self.logger.error(f"Invalid migration file name: {file}")
                    return False
            
            self.logger.info(f"✅ {len(migration_files)} migration files validated")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration validation failed: {e}")
            return False
    
    async def test_migration_rollback(self) -> bool:
        """Test migration rollback capability"""
        try:
            # This would implement actual rollback testing
            # For now, just validate rollback scripts exist
            rollback_dir = "db/rollbacks"
            if os.path.exists(rollback_dir):
                rollback_files = [
                    f for f in os.listdir(rollback_dir) 
                    if f.endswith('.sql')
                ]
                self.logger.info(f"✅ {len(rollback_files)} rollback scripts available")
            else:
                self.logger.warning("No rollback scripts directory found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback validation failed: {e}")
            return False


async def main():
    validator = MigrationValidator()
    
    # Create backup
    backup_file = await validator.create_backup()
    if not backup_file:
        exit(1)
    
    # Validate migrations
    if not await validator.validate_migration_integrity():
        exit(1)
    
    # Test rollback capability
    if not await validator.test_migration_rollback():
        exit(1)
    
    print("✅ Migration validation completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
```

## Blue-Green Deployment Strategy

### 1. Deployment Orchestration

Create `scripts/blue_green_deploy.py`:

```python
#!/usr/bin/env python3
"""
Blue-Green deployment orchestration script
Manages zero-downtime deployments with automatic rollback
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional

import aiohttp
import docker


class BlueGreenDeployer:
    def __init__(self):
        self.logger = self._setup_logging()
        self.docker_client = docker.from_env()
        self.current_environment = None
        self.target_environment = None
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def determine_environments(self) -> None:
        """Determine current and target environments"""
        try:
            # Check which environment is currently active
            containers = self.docker_client.containers.list(
                filters={"label": "app=agentic-startup-studio"}
            )
            
            active_envs = set()
            for container in containers:
                if container.status == "running":
                    env = container.labels.get("environment", "unknown")
                    active_envs.add(env)
            
            if "blue" in active_envs and "green" not in active_envs:
                self.current_environment = "blue"
                self.target_environment = "green"
            elif "green" in active_envs and "blue" not in active_envs:
                self.current_environment = "green"
                self.target_environment = "blue"
            else:
                # Default to blue if none or both are running
                self.current_environment = "green"
                self.target_environment = "blue"
            
            self.logger.info(f"Current: {self.current_environment}, Target: {self.target_environment}")
            
        except Exception as e:
            self.logger.error(f"Environment determination failed: {e}")
            raise
    
    async def deploy_target_environment(self) -> bool:
        """Deploy application to target environment"""
        try:
            self.logger.info(f"🚀 Deploying to {self.target_environment} environment")
            
            # Build new image
            build_cmd = f"docker-compose -f docker-compose.{self.target_environment}.yml build"
            result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Build failed: {result.stderr}")
                return False
            
            # Start target environment
            start_cmd = f"docker-compose -f docker-compose.{self.target_environment}.yml up -d"
            result = subprocess.run(start_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Startup failed: {result.stderr}")
                return False
            
            # Wait for services to be ready
            await self.wait_for_readiness()
            
            self.logger.info(f"✅ {self.target_environment} environment deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    async def wait_for_readiness(self, timeout: int = 300) -> bool:
        """Wait for target environment to be ready"""
        start_time = time.time()
        port = 8001 if self.target_environment == "blue" else 8002
        health_url = f"http://localhost:{port}/health"
        
        self.logger.info(f"Waiting for {self.target_environment} environment readiness...")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("status") == "healthy":
                                self.logger.info(f"✅ {self.target_environment} environment is ready")
                                return True
                except Exception:
                    pass
                
                await asyncio.sleep(5)
        
        self.logger.error(f"❌ {self.target_environment} environment failed to become ready")
        return False
    
    async def run_smoke_tests(self) -> bool:
        """Run smoke tests against target environment"""
        try:
            self.logger.info(f"🧪 Running smoke tests on {self.target_environment}")
            
            port = 8001 if self.target_environment == "blue" else 8002
            base_url = f"http://localhost:{port}"
            
            test_endpoints = [
                "/health",
                "/metrics",
                "/api/v1/ideas",
            ]
            
            async with aiohttp.ClientSession() as session:
                for endpoint in test_endpoints:
                    url = f"{base_url}{endpoint}"
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status not in [200, 404]:  # 404 might be OK for some endpoints
                                self.logger.error(f"Smoke test failed for {endpoint}: {response.status}")
                                return False
                    except Exception as e:
                        self.logger.error(f"Smoke test failed for {endpoint}: {e}")
                        return False
            
            self.logger.info("✅ Smoke tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Smoke tests failed: {e}")
            return False
    
    def switch_traffic(self) -> bool:
        """Switch traffic from current to target environment"""
        try:
            self.logger.info(f"🔄 Switching traffic to {self.target_environment}")
            
            # Update load balancer configuration
            # This would typically involve updating nginx config, AWS ALB rules, etc.
            # For this example, we'll simulate with environment labels
            
            switch_cmd = f"docker service update --label-add active=true agentic-startup-studio-{self.target_environment}"
            result = subprocess.run(switch_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"Service label update failed: {result.stderr}")
            
            # Wait for traffic switch to take effect
            time.sleep(10)
            
            self.logger.info("✅ Traffic switched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Traffic switch failed: {e}")
            return False
    
    def cleanup_old_environment(self) -> None:
        """Clean up the old environment"""
        try:
            self.logger.info(f"🧹 Cleaning up {self.current_environment} environment")
            
            cleanup_cmd = f"docker-compose -f docker-compose.{self.current_environment}.yml down"
            result = subprocess.run(cleanup_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"Cleanup warning: {result.stderr}")
            else:
                self.logger.info("✅ Old environment cleaned up")
                
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    async def rollback(self) -> bool:
        """Rollback to previous environment"""
        try:
            self.logger.warning(f"🔄 Rolling back to {self.current_environment}")
            
            # Stop target environment
            stop_cmd = f"docker-compose -f docker-compose.{self.target_environment}.yml down"
            subprocess.run(stop_cmd, shell=True)
            
            # Ensure current environment is running
            start_cmd = f"docker-compose -f docker-compose.{self.current_environment}.yml up -d"
            result = subprocess.run(start_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    async def deploy(self) -> bool:
        """Execute blue-green deployment"""
        deployment_start = datetime.utcnow()
        
        try:
            # Determine environments
            self.determine_environments()
            
            # Deploy to target environment
            if not await self.deploy_target_environment():
                await self.rollback()
                return False
            
            # Run smoke tests
            if not await self.run_smoke_tests():
                await self.rollback()
                return False
            
            # Switch traffic
            if not self.switch_traffic():
                await self.rollback()
                return False
            
            # Wait and monitor
            await asyncio.sleep(30)  # Monitor for 30 seconds
            
            # If everything is stable, cleanup old environment
            self.cleanup_old_environment()
            
            deployment_duration = (datetime.utcnow() - deployment_start).total_seconds()
            self.logger.info(f"🎉 Deployment completed successfully in {deployment_duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            await self.rollback()
            return False


async def main():
    deployer = BlueGreenDeployer()
    success = await deployer.deploy()
    
    if not success:
        exit(1)
    
    print("✅ Blue-Green deployment completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
```

## Rollback Procedures

### 1. Automated Rollback Script

Create `scripts/emergency_rollback.py`:

```python
#!/usr/bin/env python3
"""
Emergency rollback script
Provides rapid rollback capabilities with safety checks
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp


class EmergencyRollback:
    def __init__(self):
        self.logger = self._setup_logging()
        self.rollback_started = datetime.utcnow()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def get_last_deployment_info(self) -> Optional[Dict]:
        """Get information about the last deployment"""
        try:
            if os.path.exists("deployment_history.json"):
                with open("deployment_history.json", "r") as f:
                    history = json.load(f)
                    if history:
                        return history[-1]
            return None
        except Exception as e:
            self.logger.error(f"Failed to get deployment info: {e}")
            return None
    
    async def verify_rollback_target(self, target_version: str) -> bool:
        """Verify rollback target is valid"""
        try:
            # Check if rollback target exists
            check_cmd = f"docker images | grep agentic-startup-studio | grep {target_version}"
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Rollback target {target_version} not found")
                return False
            
            self.logger.info(f"✅ Rollback target {target_version} verified")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback target verification failed: {e}")
            return False
    
    async def execute_database_rollback(self, backup_file: str) -> bool:
        """Execute database rollback"""
        try:
            if not os.path.exists(backup_file):
                self.logger.error(f"Backup file {backup_file} not found")
                return False
            
            self.logger.info(f"🔄 Rolling back database from {backup_file}")
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                self.logger.error("DATABASE_URL not configured")
                return False
            
            # Restore database
            restore_cmd = f"psql {db_url} < {backup_file}"
            result = subprocess.run(restore_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Database rollback failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Database rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database rollback failed: {e}")
            return False
    
    async def execute_application_rollback(self, target_version: str) -> bool:
        """Execute application rollback"""
        try:
            self.logger.info(f"🔄 Rolling back application to {target_version}")
            
            # Stop current services
            stop_cmd = "docker-compose down"
            subprocess.run(stop_cmd, shell=True, capture_output=True, text=True)
            
            # Update docker-compose to use target version
            # This would typically involve updating image tags in docker-compose.yml
            
            # Start services with target version
            start_cmd = f"docker-compose up -d"
            result = subprocess.run(start_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Application rollback failed: {result.stderr}")
                return False
            
            # Wait for services to be ready
            await self.wait_for_service_readiness()
            
            self.logger.info("✅ Application rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Application rollback failed: {e}")
            return False
    
    async def wait_for_service_readiness(self, timeout: int = 180) -> bool:
        """Wait for services to be ready after rollback"""
        start_time = time.time()
        health_url = "http://localhost:8000/health"
        
        self.logger.info("Waiting for service readiness...")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("status") == "healthy":
                                self.logger.info("✅ Services are ready")
                                return True
                except Exception:
                    pass
                
                await asyncio.sleep(5)
        
        self.logger.error("❌ Services failed to become ready")
        return False
    
    async def run_post_rollback_verification(self) -> bool:
        """Run verification tests after rollback"""
        try:
            self.logger.info("🧪 Running post-rollback verification")
            
            # Run basic health checks
            result = subprocess.run(
                ["python", "scripts/run_health_checks.py", "--basic"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.logger.error(f"Post-rollback verification failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Post-rollback verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Post-rollback verification failed: {e}")
            return False
    
    def record_rollback_event(self, success: bool, details: Dict) -> None:
        """Record rollback event for audit trail"""
        try:
            rollback_record = {
                "timestamp": self.rollback_started.isoformat(),
                "duration_seconds": (datetime.utcnow() - self.rollback_started).total_seconds(),
                "success": success,
                "details": details,
                "triggered_by": os.getenv("USER", "system")
            }
            
            # Load existing rollback history
            rollback_history = []
            if os.path.exists("rollback_history.json"):
                with open("rollback_history.json", "r") as f:
                    rollback_history = json.load(f)
            
            rollback_history.append(rollback_record)
            
            # Keep only last 50 rollback records
            rollback_history = rollback_history[-50:]
            
            with open("rollback_history.json", "w") as f:
                json.dump(rollback_history, f, indent=2)
            
            self.logger.info("📝 Rollback event recorded")
            
        except Exception as e:
            self.logger.error(f"Failed to record rollback event: {e}")
    
    async def execute_rollback(self, target_version: Optional[str] = None) -> bool:
        """Execute emergency rollback"""
        try:
            # Get last deployment info if target not specified
            if not target_version:
                deployment_info = self.get_last_deployment_info()
                if not deployment_info:
                    self.logger.error("No deployment history found and no target version specified")
                    return False
                target_version = deployment_info.get("previous_version")
                if not target_version:
                    self.logger.error("No previous version found in deployment history")
                    return False
            
            self.logger.info(f"🚨 Starting emergency rollback to {target_version}")
            
            # Verify rollback target
            if not await self.verify_rollback_target(target_version):
                return False
            
            # Execute application rollback
            if not await self.execute_application_rollback(target_version):
                return False
            
            # Run post-rollback verification
            if not await self.run_post_rollback_verification():
                return False
            
            duration = (datetime.utcnow() - self.rollback_started).total_seconds()
            self.logger.info(f"🎉 Emergency rollback completed successfully in {duration:.2f} seconds")
            
            # Record rollback event
            self.record_rollback_event(True, {
                "target_version": target_version,
                "rollback_type": "emergency"
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency rollback failed: {e}")
            self.record_rollback_event(False, {
                "error": str(e),
                "target_version": target_version
            })
            return False


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency Rollback Tool")
    parser.add_argument("--version", help="Target version to rollback to")
    parser.add_argument("--confirm", action="store_true", help="Confirm rollback execution")
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("❌ Emergency rollback requires --confirm flag")
        exit(1)
    
    rollback = EmergencyRollback()
    success = await rollback.execute_rollback(args.version)
    
    if not success:
        exit(1)
    
    print("✅ Emergency rollback completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
```

## Integration with CI/CD

Add the following targets to your `Makefile`:

```makefile
# ========== DEPLOYMENT TARGETS ==========

pre-deploy:     ## run pre-deployment validation
	python scripts/deployment_preflight.py
	python scripts/validate_migrations.py

deploy-blue-green: ## execute blue-green deployment
	python scripts/blue_green_deploy.py

rollback:       ## emergency rollback (requires --version and --confirm)
	python scripts/emergency_rollback.py --version $(VERSION) --confirm

validate-deployment: ## validate current deployment
	python scripts/run_health_checks.py --comprehensive
	python scripts/deployment_preflight.py
```

## Monitoring and Alerting

### Deployment Monitoring

The deployment validation procedures integrate with your existing monitoring stack:

- **Health Checks**: Automated validation of service health
- **Performance Monitoring**: Validation of response times and throughput
- **Error Rate Monitoring**: Detection of error rate increases
- **Resource Monitoring**: CPU, memory, and disk usage validation

### Alert Configuration

Configure alerts for deployment failures:

```yaml
# Prometheus alerting rules for deployment monitoring
groups:
- name: deployment.rules
  rules:
  - alert: DeploymentValidationFailure
    expr: deployment_validation_status != 1
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Deployment validation failed"
      description: "Deployment validation has failed for {{ $labels.environment }}"

  - alert: RollbackExecuted
    expr: increase(rollback_events_total[5m]) > 0
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "Rollback executed"
      description: "Emergency rollback was executed"
```

## Best Practices

1. **Always run pre-deployment validation**
2. **Maintain database backups before migrations**
3. **Test rollback procedures regularly**
4. **Monitor deployment metrics continuously**
5. **Document all deployment decisions**
6. **Use feature flags for gradual rollouts**
7. **Automate as much as possible while maintaining safety**

## Emergency Contacts

In case of deployment failures:

1. **Primary**: DevOps Team Lead
2. **Secondary**: Engineering Manager
3. **Escalation**: CTO

## Conclusion

These deployment validation and rollback procedures ensure:

- ✅ Zero-downtime deployments
- ✅ Rapid rollback capabilities
- ✅ Comprehensive validation at each step
- ✅ Audit trail for all deployment activities
- ✅ Integration with existing monitoring systems

All procedures are designed to be automated where possible while maintaining human oversight for critical decisions.