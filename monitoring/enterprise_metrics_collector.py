#!/usr/bin/env python3
"""
Enterprise Metrics Collection and Analysis for Agentic Startup Studio

This module provides comprehensive metrics collection, analysis, and reporting
for enterprise environments with advanced aggregation, alerting, and compliance features.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os
from pathlib import Path
import httpx
import psutil
import aiofiles
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from contextlib import asynccontextmanager


class MetricType(Enum):
    """Types of metrics collected."""
    BUSINESS = "business"
    TECHNICAL = "technical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    USER_EXPERIENCE = "user_experience"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: Union[int, float]
    labels: Dict[str, str]
    metadata: Dict[str, Any] = None


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    description: str
    runbook_url: Optional[str] = None
    is_active: bool = False
    last_triggered: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None


class EnterpriseMetricsCollector:
    """Enterprise-grade metrics collection and monitoring."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.registry = CollectorRegistry()
        self.logger = self._setup_logging()
        
        # Prometheus metrics
        self.request_counter = Counter(
            'agentic_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'agentic_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.system_metrics = {
            'cpu_usage': Gauge('agentic_cpu_usage_percent', 'CPU usage percentage', registry=self.registry),
            'memory_usage': Gauge('agentic_memory_usage_bytes', 'Memory usage in bytes', registry=self.registry),
            'disk_usage': Gauge('agentic_disk_usage_percent', 'Disk usage percentage', registry=self.registry),
            'active_connections': Gauge('agentic_active_connections', 'Number of active connections', registry=self.registry)
        }
        
        self.business_metrics = {
            'ideas_processed': Counter('agentic_ideas_processed_total', 'Total ideas processed', registry=self.registry),
            'successful_validations': Counter('agentic_successful_validations_total', 'Successful idea validations', registry=self.registry),
            'campaign_conversions': Counter('agentic_campaign_conversions_total', 'Campaign conversions', registry=self.registry),
            'revenue_generated': Gauge('agentic_revenue_usd', 'Revenue generated in USD', registry=self.registry)
        }
        
        self.security_metrics = {
            'auth_failures': Counter('agentic_auth_failures_total', 'Authentication failures', registry=self.registry),
            'rate_limit_hits': Counter('agentic_rate_limit_hits_total', 'Rate limit violations', registry=self.registry),
            'security_alerts': Counter('agentic_security_alerts_total', 'Security alerts triggered', registry=self.registry)
        }
        
        # Data storage
        self.metrics_buffer: List[MetricDataPoint] = []
        self.alerts: Dict[str, Alert] = {}
        self.last_collection_time = datetime.now(timezone.utc)
        
        # Load alerts configuration
        self._load_alerts()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "collection_interval": 30,
            "buffer_size": 1000,
            "storage_backend": "prometheus",
            "alert_webhook_url": None,
            "compliance_mode": False,
            "data_retention_days": 90,
            "high_cardinality_metrics": False,
            "export_formats": ["prometheus", "json"],
            "enterprise_features": {
                "anomaly_detection": True,
                "predictive_alerting": True,
                "cost_tracking": True,
                "sla_monitoring": True
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for metrics collection."""
        logger = logging.getLogger("enterprise_metrics")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_alerts(self):
        """Load alert configurations."""
        default_alerts = [
            Alert(
                name="high_cpu_usage",
                condition="cpu_usage > threshold",
                severity=AlertSeverity.HIGH,
                threshold=85.0,
                description="CPU usage is above 85%",
                runbook_url="https://docs.terragonlabs.com/runbooks/high-cpu"
            ),
            Alert(
                name="high_memory_usage",
                condition="memory_usage > threshold",
                severity=AlertSeverity.HIGH,
                threshold=90.0,
                description="Memory usage is above 90%",
                runbook_url="https://docs.terragonlabs.com/runbooks/high-memory"
            ),
            Alert(
                name="auth_failure_spike",
                condition="auth_failures_rate > threshold",
                severity=AlertSeverity.CRITICAL,
                threshold=10.0,
                description="Authentication failure rate spike detected",
                runbook_url="https://docs.terragonlabs.com/runbooks/auth-failures"
            ),
            Alert(
                name="low_idea_processing_rate",
                condition="ideas_processed_rate < threshold",
                severity=AlertSeverity.MEDIUM,
                threshold=1.0,
                description="Idea processing rate is below expected threshold",
                runbook_url="https://docs.terragonlabs.com/runbooks/low-processing-rate"
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.name] = alert
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'].set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics['disk_usage'].set(disk_percent)
            
            # Network connections (simplified)
            connections = len(psutil.net_connections())
            self.system_metrics['active_connections'].set(connections)
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage_mb': memory.used / (1024 * 1024),
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk_percent,
                'active_connections': connections
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        try:
            metrics = {}
            
            # Database connection metrics (if available)
            try:
                # This would connect to actual database in real implementation
                metrics['db_connection_pool_size'] = 10
                metrics['db_active_connections'] = 3
                metrics['db_query_duration_avg'] = 0.025
            except Exception:
                pass
            
            # Cache metrics (if available)
            try:
                # This would connect to Redis/cache in real implementation
                metrics['cache_hit_rate'] = 0.85
                metrics['cache_memory_usage'] = 256
            except Exception:
                pass
            
            # API metrics
            metrics['api_response_time_p95'] = 0.450
            metrics['api_error_rate'] = 0.02
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
            return {}
    
    async def collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business and KPI metrics."""
        try:
            # In real implementation, these would come from database queries
            current_hour = datetime.now(timezone.utc).hour
            
            # Simulate business metrics based on time of day
            base_ideas = 10 + (current_hour % 12)
            base_validations = int(base_ideas * 0.7)
            base_conversions = int(base_validations * 0.3)
            
            metrics = {
                'ideas_processed_hourly': base_ideas,
                'successful_validations_hourly': base_validations,
                'campaign_conversions_hourly': base_conversions,
                'avg_validation_score': 0.72 + (current_hour % 10) * 0.02,
                'revenue_per_hour': base_conversions * 25.50,
                'customer_acquisition_cost': 45.30,
                'lifetime_value': 127.80
            }
            
            # Update Prometheus counters
            self.business_metrics['ideas_processed']._value._value += metrics['ideas_processed_hourly']
            self.business_metrics['successful_validations']._value._value += metrics['successful_validations_hourly']
            self.business_metrics['campaign_conversions']._value._value += metrics['campaign_conversions_hourly']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect business metrics: {e}")
            return {}
    
    async def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        try:
            # In real implementation, these would come from security logs/systems
            metrics = {
                'auth_failures_last_hour': 2,
                'rate_limit_violations': 0,
                'suspicious_requests': 1,
                'blocked_ips': 3,
                'security_scan_score': 95.5,
                'vulnerability_count': 0,
                'cert_expiry_days': 87
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect security metrics: {e}")
            return {}
    
    async def evaluate_alerts(self, metrics: Dict[str, Any]):
        """Evaluate alert conditions against current metrics."""
        current_time = datetime.now(timezone.utc)
        
        for alert_name, alert in self.alerts.items():
            try:
                # Skip if alert is suppressed
                if alert.suppressed_until and current_time < alert.suppressed_until:
                    continue
                
                # Evaluate condition based on alert type
                triggered = False
                
                if alert_name == "high_cpu_usage" and "cpu_usage" in metrics:
                    triggered = metrics["cpu_usage"] > alert.threshold
                
                elif alert_name == "high_memory_usage" and "memory_usage_percent" in metrics:
                    triggered = metrics["memory_usage_percent"] > alert.threshold
                
                elif alert_name == "auth_failure_spike" and "auth_failures_last_hour" in metrics:
                    triggered = metrics["auth_failures_last_hour"] > alert.threshold
                
                elif alert_name == "low_idea_processing_rate" and "ideas_processed_hourly" in metrics:
                    triggered = metrics["ideas_processed_hourly"] < alert.threshold
                
                # Handle alert state changes
                if triggered and not alert.is_active:
                    # New alert
                    alert.is_active = True
                    alert.last_triggered = current_time
                    await self._send_alert(alert, metrics)
                    self.logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
                
                elif not triggered and alert.is_active:
                    # Alert resolved
                    alert.is_active = False
                    await self._send_alert_resolution(alert)
                    self.logger.info(f"Alert resolved: {alert.name}")
            
            except Exception as e:
                self.logger.error(f"Failed to evaluate alert {alert_name}: {e}")
    
    async def _send_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Send alert notification."""
        webhook_url = self.config.get("alert_webhook_url")
        if not webhook_url:
            return
        
        try:
            payload = {
                "alert_name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "runbook_url": alert.runbook_url,
                "metrics": metrics
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Failed to send alert for {alert.name}: {e}")
    
    async def _send_alert_resolution(self, alert: Alert):
        """Send alert resolution notification."""
        webhook_url = self.config.get("alert_webhook_url")
        if not webhook_url:
            return
        
        try:
            payload = {
                "alert_name": alert.name,
                "status": "resolved",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"Alert {alert.name} has been resolved"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Failed to send alert resolution for {alert.name}: {e}")
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics from all sources."""
        start_time = time.time()
        
        try:
            # Collect metrics concurrently
            system_task = asyncio.create_task(self.collect_system_metrics())
            app_task = asyncio.create_task(self.collect_application_metrics())
            business_task = asyncio.create_task(self.collect_business_metrics())
            security_task = asyncio.create_task(self.collect_security_metrics())
            
            # Wait for all collections to complete
            system_metrics = await system_task
            app_metrics = await app_task
            business_metrics = await business_task
            security_metrics = await security_task
            
            # Combine all metrics
            all_metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_duration_seconds": time.time() - start_time,
                "system": system_metrics,
                "application": app_metrics,
                "business": business_metrics,
                "security": security_metrics
            }
            
            # Evaluate alerts
            flat_metrics = {**system_metrics, **app_metrics, **business_metrics, **security_metrics}
            await self.evaluate_alerts(flat_metrics)
            
            # Store metrics
            await self._store_metrics(all_metrics)
            
            self.logger.info(f"Metrics collection completed in {all_metrics['collection_duration_seconds']:.3f}s")
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics to configured backends."""
        try:
            # Store to file (always enabled for backup)
            metrics_dir = Path("metrics_data")
            metrics_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
            metrics_file = metrics_dir / f"metrics_{timestamp}.jsonl"
            
            async with aiofiles.open(metrics_file, "a") as f:
                await f.write(json.dumps(metrics) + "\n")
            
            # Store to other backends if configured
            storage_backend = self.config.get("storage_backend")
            if storage_backend == "prometheus":
                # Prometheus metrics are automatically collected via registry
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def start_collection_loop(self, interval: Optional[int] = None):
        """Start continuous metrics collection loop."""
        collection_interval = interval or self.config.get("collection_interval", 30)
        
        self.logger.info(f"Starting metrics collection loop (interval: {collection_interval}s)")
        
        try:
            while True:
                await self.collect_all_metrics()
                await asyncio.sleep(collection_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Metrics collection stopped by user")
        except Exception as e:
            self.logger.error(f"Metrics collection loop failed: {e}")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for dashboard visualization."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": {
                "status": "healthy",  # Would be calculated based on metrics
                "cpu_usage": self.system_metrics['cpu_usage']._value._value,
                "memory_usage": self.system_metrics['memory_usage']._value._value,
                "disk_usage": self.system_metrics['disk_usage']._value._value
            },
            "business_kpis": {
                "ideas_processed": self.business_metrics['ideas_processed']._value._value,
                "success_rate": 0.72,  # Would be calculated
                "revenue": self.business_metrics['revenue_generated']._value._value
            },
            "active_alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "description": alert.description
                }
                for alert in self.alerts.values()
                if alert.is_active
            ]
        }


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Metrics Collector")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--interval", type=int, default=30, help="Collection interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--prometheus", action="store_true", help="Output Prometheus metrics")
    
    args = parser.parse_args()
    
    collector = EnterpriseMetricsCollector(args.config)
    
    if args.prometheus:
        # Output Prometheus metrics and exit
        metrics = collector.get_prometheus_metrics()
        print(metrics)
        return
    
    if args.once:
        # Collect metrics once and exit
        result = await collector.collect_all_metrics()
        print(json.dumps(result, indent=2))
    else:
        # Start continuous collection
        await collector.start_collection_loop(args.interval)


if __name__ == "__main__":
    asyncio.run(main())