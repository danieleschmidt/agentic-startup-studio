#!/usr/bin/env python3
"""
Production Deployment Readiness Check
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

class DeploymentChecker:
    def __init__(self):
        self.results = {
            "container_config": {},
            "environment_config": {},
            "security_config": {},
            "monitoring_config": {},
            "scaling_config": {},
            "backup_config": {}
        }
        self.passed_checks = 0
        self.total_checks = 0
    
    def check_file(self, file_path, description=""):
        """Check if a file exists and log result."""
        self.total_checks += 1
        exists = Path(file_path).exists()
        if exists:
            self.passed_checks += 1
        
        status = "✅ PASS" if exists else "❌ MISS"
        print(f"  {status} {description}: {file_path}")
        return exists
    
    def check_container_configuration(self):
        """Check Docker and container configuration."""
        print("\n🐳 CONTAINER CONFIGURATION")
        
        # Check for Docker files
        dockerfile_exists = self.check_file("Dockerfile", "Main Dockerfile")
        compose_exists = self.check_file("docker-compose.yml", "Docker Compose")
        compose_prod_exists = self.check_file("docker-compose.prod.yml", "Production Compose")
        
        # Check for Kubernetes files
        k8s_dir_exists = Path("k8s").is_dir() or Path("kubernetes").is_dir()
        if k8s_dir_exists:
            self.passed_checks += 1
            print("  ✅ PASS Kubernetes configurations found")
        else:
            print("  ⚠️  MISS Kubernetes configurations")
        self.total_checks += 1
        
        # Check for build optimizations
        dockerignore_exists = self.check_file(".dockerignore", "Docker ignore file")
        
        self.results["container_config"] = {
            "dockerfile": dockerfile_exists,
            "docker_compose": compose_exists,
            "production_compose": compose_prod_exists,
            "kubernetes": k8s_dir_exists,
            "dockerignore": dockerignore_exists
        }
    
    def check_environment_configuration(self):
        """Check environment and configuration management."""
        print("\n🌍 ENVIRONMENT CONFIGURATION")
        
        # Check for environment files
        env_example_exists = self.check_file(".env.example", "Environment template")
        env_prod_exists = self.check_file(".env.prod", "Production environment") or self.check_file(".env.production", "Production environment")
        
        # Check configuration files
        settings_exists = self.check_file("pipeline/config/settings.py", "Settings module")
        
        # Check for secrets management
        secrets_exists = self.check_file("pipeline/config/secrets_manager.py", "Secrets manager")
        
        # Check for startup scripts
        startup_exists = self.check_file("start.sh", "Startup script") or self.check_file("scripts/start.sh", "Startup script")
        
        self.results["environment_config"] = {
            "env_example": env_example_exists,
            "env_production": env_prod_exists,
            "settings_module": settings_exists,
            "secrets_manager": secrets_exists,
            "startup_script": startup_exists
        }
    
    def check_security_configuration(self):
        """Check security-related configurations."""
        print("\n🔒 SECURITY CONFIGURATION")
        
        # Check SSL/TLS configuration
        try:
            from pipeline.config.settings import get_settings
            settings = get_settings()
            
            ssl_enabled = hasattr(settings.database, 'enable_ssl') and settings.database.enable_ssl
            if ssl_enabled:
                self.passed_checks += 1
                print("  ✅ PASS Database SSL enabled")
            else:
                print("  ❌ MISS Database SSL not enabled")
            self.total_checks += 1
            
            # Check CORS configuration
            cors_configured = hasattr(settings, 'allowed_origins')
            if cors_configured:
                self.passed_checks += 1
                print("  ✅ PASS CORS configuration found")
            else:
                print("  ❌ MISS CORS configuration missing")
            self.total_checks += 1
            
            # Check secret key configuration
            secret_key_configured = hasattr(settings, 'secret_key') and settings.secret_key
            if secret_key_configured:
                self.passed_checks += 1
                print("  ✅ PASS Secret key configured")
            else:
                print("  ❌ MISS Secret key missing")
            self.total_checks += 1
            
        except Exception as e:
            print(f"  ❌ MISS Settings import failed: {e}")
            self.total_checks += 3
        
        # Check for security headers configuration
        nginx_config_exists = self.check_file("nginx.conf", "Nginx configuration") or self.check_file("configs/nginx.conf", "Nginx configuration")
        
        self.results["security_config"] = {
            "database_ssl": ssl_enabled if 'ssl_enabled' in locals() else False,
            "cors_configured": cors_configured if 'cors_configured' in locals() else False,
            "secret_key": secret_key_configured if 'secret_key_configured' in locals() else False,
            "nginx_config": nginx_config_exists
        }
    
    def check_monitoring_configuration(self):
        """Check monitoring and observability configuration."""
        print("\n📊 MONITORING CONFIGURATION")
        
        # Check for monitoring files
        prometheus_exists = self.check_file("prometheus.yml", "Prometheus config") or self.check_file("configs/prometheus.yml", "Prometheus config")
        grafana_exists = self.check_file("grafana/", "Grafana config") or Path("grafana").is_dir()
        
        if grafana_exists:
            self.passed_checks += 1
            print("  ✅ PASS Grafana configuration directory found")
        else:
            print("  ❌ MISS Grafana configuration directory")
        self.total_checks += 1
        
        # Check for health endpoints
        try:
            health_server_exists = Path("pipeline/api/health_server.py").exists()
            if health_server_exists:
                self.passed_checks += 1
                print("  ✅ PASS Health check endpoints found")
            else:
                print("  ❌ MISS Health check endpoints")
            self.total_checks += 1
        except Exception:
            self.total_checks += 1
        
        # Check for logging configuration
        try:
            from pipeline.config.settings import get_logging_config
            logging_config = get_logging_config()
            
            structured_logging = hasattr(logging_config, 'enable_json_logging') and logging_config.enable_json_logging
            if structured_logging:
                self.passed_checks += 1
                print("  ✅ PASS Structured logging configured")
            else:
                print("  ❌ MISS Structured logging not configured")
            self.total_checks += 1
            
        except Exception:
            print("  ❌ MISS Logging configuration not accessible")
            self.total_checks += 1
        
        self.results["monitoring_config"] = {
            "prometheus": prometheus_exists,
            "grafana": grafana_exists,
            "health_checks": health_server_exists if 'health_server_exists' in locals() else False,
            "structured_logging": structured_logging if 'structured_logging' in locals() else False
        }
    
    def check_scaling_configuration(self):
        """Check horizontal and vertical scaling readiness."""
        print("\n⚡ SCALING CONFIGURATION")
        
        # Check for load balancer configuration
        load_balancer_exists = self.check_file("load-balancer.conf", "Load balancer config") or self.check_file("configs/haproxy.cfg", "HAProxy config")
        
        # Check for auto-scaling configuration
        hpa_exists = self.check_file("k8s/hpa.yml", "Horizontal Pod Autoscaler") or self.check_file("kubernetes/hpa.yaml", "HPA")
        
        # Check for service mesh configuration
        istio_exists = self.check_file("istio/", "Istio configuration") or Path("istio").is_dir()
        if istio_exists:
            self.passed_checks += 1
            print("  ✅ PASS Service mesh configuration found")
        else:
            print("  ⚠️  INFO Service mesh not configured (optional)")
        self.total_checks += 1
        
        # Check database connection pooling
        try:
            from pipeline.config.settings import get_db_config
            db_config = get_db_config()
            
            pooling_configured = hasattr(db_config, 'max_connections') and db_config.max_connections >= 20
            if pooling_configured:
                self.passed_checks += 1
                print("  ✅ PASS Database connection pooling configured")
            else:
                print("  ❌ MISS Database connection pooling insufficient")
            self.total_checks += 1
        except Exception:
            print("  ❌ MISS Database configuration not accessible")
            self.total_checks += 1
        
        self.results["scaling_config"] = {
            "load_balancer": load_balancer_exists,
            "horizontal_scaling": hpa_exists,
            "service_mesh": istio_exists,
            "connection_pooling": pooling_configured if 'pooling_configured' in locals() else False
        }
    
    def check_backup_configuration(self):
        """Check backup and disaster recovery configuration."""
        print("\n💾 BACKUP & DISASTER RECOVERY")
        
        # Check for backup scripts
        backup_script_exists = self.check_file("scripts/backup.sh", "Backup script") or self.check_file("backup.sh", "Backup script")
        
        # Check for database migration scripts
        migrations_exist = Path("migrations").is_dir() or Path("alembic").is_dir()
        if migrations_exist:
            self.passed_checks += 1
            print("  ✅ PASS Database migrations found")
        else:
            print("  ❌ MISS Database migrations directory")
        self.total_checks += 1
        
        # Check for disaster recovery documentation
        dr_docs_exist = self.check_file("docs/disaster-recovery.md", "DR documentation") or self.check_file("DISASTER_RECOVERY.md", "DR docs")
        
        # Check for data retention policies
        retention_config_exists = self.check_file("configs/retention.yml", "Data retention config")
        
        self.results["backup_config"] = {
            "backup_scripts": backup_script_exists,
            "database_migrations": migrations_exist,
            "disaster_recovery_docs": dr_docs_exist,
            "retention_policies": retention_config_exists
        }
    
    def generate_deployment_summary(self):
        """Generate deployment readiness summary."""
        print("\n" + "=" * 60)
        print("🚀 PRODUCTION DEPLOYMENT READINESS SUMMARY")
        print("=" * 60)
        
        pass_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"Deployment Checks: {self.passed_checks}/{self.total_checks} ({pass_rate:.1f}%)")
        
        if pass_rate >= 90:
            status = "🎉 PRODUCTION READY"
            recommendation = "System is ready for production deployment"
        elif pass_rate >= 75:
            status = "✅ MOSTLY READY"
            recommendation = "Address minor issues before production deployment"
        elif pass_rate >= 60:
            status = "⚠️  NEEDS WORK"
            recommendation = "Significant preparation needed before production"
        else:
            status = "❌ NOT READY"
            recommendation = "Major deployment components missing"
        
        print(f"Status: {status}")
        print(f"Recommendation: {recommendation}")
        
        # Critical missing components
        print("\n🎯 DEPLOYMENT PRIORITIES:")
        
        if not self.results["container_config"].get("dockerfile", False):
            print("  🚨 HIGH: Create Dockerfile for containerization")
        
        if not self.results["security_config"].get("database_ssl", False):
            print("  🚨 HIGH: Enable database SSL/TLS")
        
        if not self.results["monitoring_config"].get("health_checks", False):
            print("  🔧 MEDIUM: Implement health check endpoints")
        
        if not self.results["backup_config"].get("database_migrations", False):
            print("  🔧 MEDIUM: Set up database migration system")
        
        print("  💡 Consider implementing CI/CD pipeline automation")
        print("  📊 Set up monitoring and alerting systems")
        print("  🔄 Plan for zero-downtime deployments")
        
        # Save results
        deployment_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "pass_rate": pass_rate,
                "status": status,
                "recommendation": recommendation
            },
            "detailed_results": self.results
        }
        
        with open('deployment_readiness_report.json', 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to deployment_readiness_report.json")
        
        return pass_rate >= 75

def main():
    """Main deployment readiness check."""
    print("🚀 PRODUCTION DEPLOYMENT READINESS CHECK")
    print("=" * 60)
    
    checker = DeploymentChecker()
    
    # Run all checks
    checker.check_container_configuration()
    checker.check_environment_configuration()
    checker.check_security_configuration()
    checker.check_monitoring_configuration()
    checker.check_scaling_configuration()
    checker.check_backup_configuration()
    
    # Generate summary
    success = checker.generate_deployment_summary()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)