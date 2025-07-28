#!/usr/bin/env python3

"""
Monitoring setup and health check script for Agentic Startup Studio.
Validates monitoring stack configuration and deployment.
"""

import os
import sys
import subprocess
import json
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class MonitoringSetup:
    """Manages monitoring stack setup and validation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = self.project_root / "monitoring"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        if level == "DEBUG" and not self.verbose:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "📊" if level == "INFO" else "🐛" if level == "DEBUG" else "⚠️" if level == "WARN" else "❌"
        print(f"{prefix} [{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], description: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and handle errors."""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}", "DEBUG")
            
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=capture_output,
                text=True
            )
            self.log(f"✅ {description} completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} failed: {e}", "ERROR")
            if capture_output and e.stdout:
                self.log(f"STDOUT: {e.stdout}", "DEBUG")
            if capture_output and e.stderr:
                self.log(f"STDERR: {e.stderr}", "DEBUG")
            raise
    
    def validate_docker_compose_files(self) -> bool:
        """Validate Docker Compose configuration files."""
        self.log("Validating Docker Compose configurations...")
        
        compose_files = [
            self.monitoring_dir / "docker-compose.monitoring.yml",
            self.project_root / "docker-compose.yml",
            self.project_root / "docker-compose.dev.yml",
        ]
        
        for compose_file in compose_files:
            if compose_file.exists():
                try:
                    self.run_command(
                        ["docker-compose", "-f", str(compose_file), "config"],
                        f"Validating {compose_file.name}"
                    )
                except subprocess.CalledProcessError:
                    self.log(f"Invalid Docker Compose file: {compose_file}", "ERROR")
                    return False
            else:
                self.log(f"Missing Docker Compose file: {compose_file}", "WARN")
        
        return True
    
    def validate_monitoring_configs(self) -> bool:
        """Validate monitoring service configurations."""
        self.log("Validating monitoring configurations...")
        
        config_files = {
            "prometheus.yml": self._validate_prometheus_config,
            "alertmanager.yml": self._validate_alertmanager_config,
            "loki-config.yml": self._validate_loki_config,
            "otel-collector-config.yml": self._validate_otel_config,
        }
        
        for config_file, validator in config_files.items():
            config_path = self.monitoring_dir / config_file
            if config_path.exists():
                try:
                    validator(config_path)
                    self.log(f"✅ {config_file} is valid")
                except Exception as e:
                    self.log(f"❌ Invalid {config_file}: {e}", "ERROR")
                    return False
            else:
                self.log(f"Missing configuration file: {config_file}", "WARN")
        
        return True
    
    def _validate_prometheus_config(self, config_path: Path) -> None:
        """Validate Prometheus configuration."""
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["global", "scrape_configs"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            # Validate scrape configs
            for scrape_config in config["scrape_configs"]:
                if "job_name" not in scrape_config:
                    raise ValueError("Scrape config missing job_name")
                    
        except ImportError:
            self.log("PyYAML not available, skipping detailed validation", "WARN")
    
    def _validate_alertmanager_config(self, config_path: Path) -> None:
        """Validate AlertManager configuration."""
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["route", "receivers"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
                    
        except ImportError:
            self.log("PyYAML not available, skipping detailed validation", "WARN")
    
    def _validate_loki_config(self, config_path: Path) -> None:
        """Validate Loki configuration."""
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["server", "schema_config"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
                    
        except ImportError:
            self.log("PyYAML not available, skipping detailed validation", "WARN")
    
    def _validate_otel_config(self, config_path: Path) -> None:
        """Validate OpenTelemetry Collector configuration."""
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["receivers", "processors", "exporters", "service"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
                    
        except ImportError:
            self.log("PyYAML not available, skipping detailed validation", "WARN")
    
    def start_monitoring_stack(self) -> bool:
        """Start the monitoring stack."""
        self.log("Starting monitoring stack...")
        
        compose_file = self.monitoring_dir / "docker-compose.monitoring.yml"
        
        try:
            # Pull latest images
            self.run_command(
                ["docker-compose", "-f", str(compose_file), "pull"],
                "Pulling monitoring images",
                capture_output=False
            )
            
            # Start services
            self.run_command(
                ["docker-compose", "-f", str(compose_file), "up", "-d"],
                "Starting monitoring services",
                capture_output=False
            )
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to start monitoring stack", "ERROR")
            return False
    
    def wait_for_services(self, timeout: int = 120) -> bool:
        """Wait for monitoring services to be ready."""
        self.log("Waiting for monitoring services to be ready...")
        
        services = {
            "Prometheus": "http://localhost:9090/-/ready",
            "Grafana": "http://localhost:3000/api/health",
            "AlertManager": "http://localhost:9093/-/ready",
            "Loki": "http://localhost:3100/ready",
        }
        
        start_time = time.time()
        
        for service_name, health_url in services.items():
            self.log(f"Waiting for {service_name}...")
            
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        self.log(f"✅ {service_name} is ready")
                        break
                except requests.RequestException:
                    pass
                
                time.sleep(5)
            else:
                self.log(f"❌ {service_name} failed to start within {timeout}s", "ERROR")
                return False
        
        return True
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run comprehensive health checks on monitoring services."""
        self.log("Running monitoring health checks...")
        
        results = {}
        
        # Prometheus health check
        results["prometheus"] = self._check_prometheus_health()
        
        # Grafana health check
        results["grafana"] = self._check_grafana_health()
        
        # AlertManager health check
        results["alertmanager"] = self._check_alertmanager_health()
        
        # Loki health check
        results["loki"] = self._check_loki_health()
        
        # Jaeger health check
        results["jaeger"] = self._check_jaeger_health()
        
        return results
    
    def _check_prometheus_health(self) -> bool:
        """Check Prometheus health and configuration."""
        try:
            # Check readiness
            response = requests.get("http://localhost:9090/-/ready", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check targets
            response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
            if response.status_code == 200:
                targets = response.json()
                active_targets = [t for t in targets["data"]["activeTargets"] if t["health"] == "up"]
                self.log(f"Prometheus: {len(active_targets)} active targets")
            
            return True
            
        except requests.RequestException:
            return False
    
    def _check_grafana_health(self) -> bool:
        """Check Grafana health and datasources."""
        try:
            # Check health
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check datasources (with basic auth)
            response = requests.get(
                "http://localhost:3000/api/datasources",
                auth=("admin", "admin"),
                timeout=5
            )
            if response.status_code == 200:
                datasources = response.json()
                self.log(f"Grafana: {len(datasources)} datasources configured")
            
            return True
            
        except requests.RequestException:
            return False
    
    def _check_alertmanager_health(self) -> bool:
        """Check AlertManager health."""
        try:
            response = requests.get("http://localhost:9093/-/ready", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _check_loki_health(self) -> bool:
        """Check Loki health."""
        try:
            response = requests.get("http://localhost:3100/ready", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _check_jaeger_health(self) -> bool:
        """Check Jaeger health."""
        try:
            response = requests.get("http://localhost:16686/api/services", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate_test_data(self) -> None:
        """Generate test data for monitoring validation."""
        self.log("Generating test data...")
        
        # Generate metrics by making requests to the application
        app_urls = [
            "http://localhost:8000/health",
            "http://localhost:8000/metrics",
            "http://localhost:8000/api/v1/ideas",
        ]
        
        for url in app_urls:
            try:
                for _ in range(5):
                    requests.get(url, timeout=1)
                    time.sleep(0.1)
                self.log(f"Generated test traffic for {url}")
            except requests.RequestException:
                self.log(f"Failed to generate test traffic for {url}", "WARN")
    
    def generate_monitoring_report(self, health_results: Dict[str, bool]) -> Path:
        """Generate monitoring setup report."""
        report_dir = self.project_root / "monitoring" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"monitoring_setup_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "health_checks": health_results,
            "summary": {
                "total_services": len(health_results),
                "healthy_services": sum(health_results.values()),
                "unhealthy_services": len(health_results) - sum(health_results.values()),
                "overall_health": all(health_results.values()),
            },
            "services": {
                "prometheus": {
                    "url": "http://localhost:9090",
                    "description": "Metrics collection and alerting",
                    "healthy": health_results.get("prometheus", False),
                },
                "grafana": {
                    "url": "http://localhost:3000",
                    "description": "Visualization and dashboards",
                    "healthy": health_results.get("grafana", False),
                },
                "alertmanager": {
                    "url": "http://localhost:9093",
                    "description": "Alert routing and management",
                    "healthy": health_results.get("alertmanager", False),
                },
                "loki": {
                    "url": "http://localhost:3100",
                    "description": "Log aggregation",
                    "healthy": health_results.get("loki", False),
                },
                "jaeger": {
                    "url": "http://localhost:16686",
                    "description": "Distributed tracing",
                    "healthy": health_results.get("jaeger", False),
                },
            },
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Monitoring report saved to {report_file}")
        return report_file
    
    def print_setup_summary(self, health_results: Dict[str, bool]) -> None:
        """Print monitoring setup summary."""
        print("\n" + "=" * 60)
        print("📊 MONITORING SETUP SUMMARY")
        print("=" * 60)
        
        for service, healthy in health_results.items():
            status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
            print(f"{service.title()}: {status}")
        
        print("\n📋 Service URLs:")
        urls = {
            "Prometheus": "http://localhost:9090",
            "Grafana": "http://localhost:3000 (admin/admin)",
            "AlertManager": "http://localhost:9093",
            "Loki": "http://localhost:3100",
            "Jaeger": "http://localhost:16686",
        }
        
        for service, url in urls.items():
            print(f"  {service}: {url}")
        
        print("\n🚀 Next Steps:")
        print("  1. Import Grafana dashboards from grafana/dashboards/")
        print("  2. Configure alert notification channels")
        print("  3. Set up log shipping from application")
        print("  4. Configure distributed tracing in application")
        print("=" * 60)
    
    def full_setup(self, start_services: bool = True) -> bool:
        """Run complete monitoring setup."""
        self.log("🚀 Starting full monitoring setup...")
        
        try:
            # 1. Validate configurations
            if not self.validate_docker_compose_files():
                return False
            
            if not self.validate_monitoring_configs():
                return False
            
            # 2. Start services if requested
            if start_services:
                if not self.start_monitoring_stack():
                    return False
                
                if not self.wait_for_services():
                    return False
            
            # 3. Run health checks
            health_results = self.run_health_checks()
            
            # 4. Generate test data
            self.generate_test_data()
            
            # 5. Generate report
            report_file = self.generate_monitoring_report(health_results)
            
            # 6. Print summary
            self.print_setup_summary(health_results)
            
            overall_success = all(health_results.values())
            if overall_success:
                self.log("🎉 Monitoring setup completed successfully!")
            else:
                self.log("⚠️ Monitoring setup completed with some issues", "WARN")
            
            return overall_success
            
        except Exception as e:
            self.log(f"Monitoring setup failed: {e}", "ERROR")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup and validate monitoring stack")
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Don't start services, just validate configurations"
    )
    parser.add_argument(
        "--health-check-only",
        action="store_true",
        help="Only run health checks on existing services"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    setup = MonitoringSetup(verbose=args.verbose)
    
    if args.health_check_only:
        health_results = setup.run_health_checks()
        setup.print_setup_summary(health_results)
        success = all(health_results.values())
    else:
        success = setup.full_setup(start_services=not args.no_start)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()