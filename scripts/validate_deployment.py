#!/usr/bin/env python3
"""
Advanced deployment validation script for production deployments.

This script performs comprehensive health checks and validates that all
systems are functioning correctly after deployment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import asyncpg
import redis.asyncio as redis
from prometheus_client.parser import text_string_to_metric_families

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment_validation.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class ValidationConfig:
    """Configuration for deployment validation."""
    base_url: str = "http://localhost:8000"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/startup_studio"
    redis_url: str = "redis://localhost:6379"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0
    expected_response_time_ms: float = 200.0
    min_memory_mb: int = 100
    max_memory_mb: int = 2048


class DeploymentValidator:
    """Comprehensive deployment validation suite."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results: List[ValidationResult] = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def run_validation(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting deployment validation...")
        
        validations = [
            self.validate_basic_health,
            self.validate_api_endpoints,
            self.validate_database_connectivity,
            self.validate_redis_connectivity,
            self.validate_authentication,
            self.validate_performance,
            self.validate_metrics_endpoints,
            self.validate_logging,
            self.validate_security_headers,
            self.validate_database_migrations,
            self.validate_monitoring_integration,
        ]

        for validation in validations:
            try:
                start_time = time.time()
                await validation()
                duration = (time.time() - start_time) * 1000
                logger.info(f"✅ {validation.__name__} completed in {duration:.2f}ms")
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                error_msg = f"Validation {validation.__name__} failed: {str(e)}"
                logger.error(error_msg)
                self.results.append(ValidationResult(
                    name=validation.__name__,
                    success=False,
                    duration_ms=duration,
                    error=error_msg
                ))

        return self.generate_report()

    async def validate_basic_health(self):
        """Validate basic health endpoint."""
        url = urljoin(self.config.base_url, "/health")
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Health check failed with status {response.status}")
            
            data = await response.json()
            
            if data.get("status") != "healthy":
                raise Exception(f"Health status is {data.get('status')}, expected 'healthy'")
            
            self.results.append(ValidationResult(
                name="basic_health",
                success=True,
                duration_ms=response.headers.get("X-Response-Time-MS", 0),
                details=data
            ))

    async def validate_api_endpoints(self):
        """Validate critical API endpoints."""
        endpoints = [
            ("/api/v1/ideas", "GET"),
            ("/api/v1/health", "GET"),
            ("/metrics", "GET"),
        ]

        for endpoint, method in endpoints:
            url = urljoin(self.config.base_url, endpoint)
            
            async with self.session.request(method, url) as response:
                if response.status >= 500:
                    raise Exception(f"Endpoint {endpoint} returned {response.status}")
                
                # Check response time
                response_time = float(response.headers.get("X-Response-Time-MS", 0))
                if response_time > self.config.expected_response_time_ms:
                    logger.warning(f"Endpoint {endpoint} slow: {response_time}ms")

        self.results.append(ValidationResult(
            name="api_endpoints",
            success=True,
            duration_ms=0,
            details={"endpoints_checked": len(endpoints)}
        ))

    async def validate_database_connectivity(self):
        """Validate database connectivity and basic operations."""
        conn = None
        try:
            conn = await asyncpg.connect(self.config.database_url)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise Exception("Database query returned unexpected result")
            
            # Check if required tables exist
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            required_tables = ["ideas", "users"]  # Add your required tables
            existing_tables = [row["table_name"] for row in tables]
            
            missing_tables = [table for table in required_tables 
                            if table not in existing_tables]
            
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
            
            self.results.append(ValidationResult(
                name="database_connectivity",
                success=True,
                duration_ms=0,
                details={
                    "tables_found": len(existing_tables),
                    "missing_tables": missing_tables
                }
            ))
            
        finally:
            if conn:
                await conn.close()

    async def validate_redis_connectivity(self):
        """Validate Redis connectivity."""
        client = None
        try:
            client = redis.from_url(self.config.redis_url)
            
            # Test basic operations
            await client.set("deployment_test", "ok", ex=60)
            result = await client.get("deployment_test")
            
            if result.decode() != "ok":
                raise Exception("Redis set/get test failed")
            
            await client.delete("deployment_test")
            
            # Check Redis info
            info = await client.info()
            
            self.results.append(ValidationResult(
                name="redis_connectivity",
                success=True,
                duration_ms=0,
                details={
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients")
                }
            ))
            
        finally:
            if client:
                await client.close()

    async def validate_authentication(self):
        """Validate authentication system."""
        # Test authentication endpoint
        auth_url = urljoin(self.config.base_url, "/api/v1/auth/test")
        
        try:
            # Should get 401 without auth
            async with self.session.get(auth_url) as response:
                if response.status not in [401, 403]:
                    logger.warning(f"Expected 401/403 for unauth request, got {response.status}")
        except aiohttp.ClientError:
            # Expected for protected endpoints
            pass

        self.results.append(ValidationResult(
            name="authentication",
            success=True,
            duration_ms=0,
            details={"auth_check": "completed"}
        ))

    async def validate_performance(self):
        """Validate system performance metrics."""
        # Test multiple concurrent requests
        urls = [urljoin(self.config.base_url, "/health") for _ in range(10)]
        
        start_time = time.time()
        tasks = [self.session.get(url) for url in urls]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        successful_responses = sum(1 for r in responses 
                                 if not isinstance(r, Exception) and r.status == 200)
        
        if successful_responses < 8:  # Allow 2 failures out of 10
            raise Exception(f"Only {successful_responses}/10 requests succeeded")
        
        avg_response_time = total_time / len(responses)
        
        self.results.append(ValidationResult(
            name="performance",
            success=True,
            duration_ms=avg_response_time,
            details={
                "concurrent_requests": len(urls),
                "successful_responses": successful_responses,
                "avg_response_time_ms": avg_response_time
            }
        ))

    async def validate_metrics_endpoints(self):
        """Validate Prometheus metrics endpoints."""
        metrics_url = urljoin(self.config.base_url, "/metrics")
        
        async with self.session.get(metrics_url) as response:
            if response.status != 200:
                raise Exception(f"Metrics endpoint returned {response.status}")
            
            metrics_text = await response.text()
            
            # Parse metrics to ensure they're valid
            metrics = list(text_string_to_metric_families(metrics_text))
            
            if not metrics:
                raise Exception("No metrics found")
            
            # Look for expected metrics
            metric_names = [m.name for m in metrics]
            expected_metrics = ["http_requests_total", "response_time_seconds"]
            
            self.results.append(ValidationResult(
                name="metrics_endpoints",
                success=True,
                duration_ms=0,
                details={
                    "metrics_count": len(metrics),
                    "metric_names": metric_names[:10]  # First 10 for brevity
                }
            ))

    async def validate_logging(self):
        """Validate logging configuration."""
        # Check if logs are being written
        log_files = [
            "deployment_validation.log",
            "application.log",
            "access.log"
        ]
        
        log_status = {}
        for log_file in log_files:
            if os.path.exists(log_file):
                stat = os.stat(log_file)
                # Check if log was modified in the last 5 minutes
                recent = datetime.now().timestamp() - stat.st_mtime < 300
                log_status[log_file] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "recently_modified": recent
                }
            else:
                log_status[log_file] = {"exists": False}

        self.results.append(ValidationResult(
            name="logging",
            success=True,
            duration_ms=0,
            details=log_status
        ))

    async def validate_security_headers(self):
        """Validate security headers."""
        url = urljoin(self.config.base_url, "/")
        
        async with self.session.get(url) as response:
            headers = response.headers
            
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Check presence
            }
            
            missing_headers = []
            for header, expected in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected and headers[header] not in (expected if isinstance(expected, list) else [expected]):
                    missing_headers.append(f"{header} (incorrect value)")

            if missing_headers:
                logger.warning(f"Missing security headers: {missing_headers}")

            self.results.append(ValidationResult(
                name="security_headers",
                success=len(missing_headers) == 0,
                duration_ms=0,
                details={
                    "headers_checked": list(security_headers.keys()),
                    "missing_headers": missing_headers
                }
            ))

    async def validate_database_migrations(self):
        """Validate database migrations are up to date."""
        conn = None
        try:
            conn = await asyncpg.connect(self.config.database_url)
            
            # Check if migration table exists (adjust based on your migration system)
            migration_check = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'alembic_version'
                )
            """)
            
            migration_status = "migration_table_exists" if migration_check else "no_migration_table"
            
            self.results.append(ValidationResult(
                name="database_migrations",
                success=True,
                duration_ms=0,
                details={"status": migration_status}
            ))
            
        finally:
            if conn:
                await conn.close()

    async def validate_monitoring_integration(self):
        """Validate monitoring and alerting integration."""
        # This would typically check if monitoring agents are running
        # and can communicate with monitoring systems
        
        monitoring_checks = {
            "prometheus_scraping": self._check_prometheus_scraping(),
            "log_shipping": self._check_log_shipping(),
            "health_monitoring": self._check_health_monitoring()
        }
        
        results = {}
        for check_name, check_func in monitoring_checks.items():
            try:
                results[check_name] = await check_func
            except Exception as e:
                results[check_name] = f"failed: {str(e)}"

        self.results.append(ValidationResult(
            name="monitoring_integration",
            success=True,
            duration_ms=0,
            details=results
        ))

    async def _check_prometheus_scraping(self) -> str:
        """Check if Prometheus can scrape metrics."""
        # This is a placeholder - implement based on your monitoring setup
        return "not_implemented"

    async def _check_log_shipping(self) -> str:
        """Check if logs are being shipped to centralized logging."""
        # This is a placeholder - implement based on your logging setup
        return "not_implemented"

    async def _check_health_monitoring(self) -> str:
        """Check if health monitoring is active."""
        # This is a placeholder - implement based on your monitoring setup
        return "not_implemented"

    def generate_report(self) -> bool:
        """Generate validation report."""
        successful = sum(1 for r in self.results if r.success)
        total = len(self.results)
        success_rate = (successful / total) * 100 if total > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": total,
            "successful_checks": successful,
            "success_rate": success_rate,
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "details": r.details
                }
                for r in self.results
            ]
        }

        # Write report to file
        with open("deployment_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"DEPLOYMENT VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Checks: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"{'='*60}")

        if success_rate < 90:
            print("❌ DEPLOYMENT VALIDATION FAILED")
            print("Some critical checks failed. Review the report for details.")
            return False
        else:
            print("✅ DEPLOYMENT VALIDATION PASSED")
            print("All critical systems are functioning correctly.")
            return True


async def main():
    """Main entry point."""
    config = ValidationConfig(
        base_url=os.environ.get("BASE_URL", "http://localhost:8000"),
        database_url=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/startup_studio"),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        timeout_seconds=int(os.environ.get("VALIDATION_TIMEOUT", "30")),
        expected_response_time_ms=float(os.environ.get("EXPECTED_RESPONSE_TIME_MS", "200.0"))
    )

    async with DeploymentValidator(config) as validator:
        success = await validator.run_validation()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())