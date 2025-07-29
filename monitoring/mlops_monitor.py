"""
MLOps Performance Monitoring for AI Model Lifecycle
Comprehensive monitoring system for AI model performance, drift detection, and optimization.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import hashlib

# Prometheus metrics (if available)
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class ModelHealth(Enum):
    """Model health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"


class DriftType(Enum):
    """Types of model drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class ModelMetrics:
    """Core model performance metrics."""
    model_name: str
    version: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    inference_time_ms: Optional[float] = None
    throughput_qps: Optional[float] = None
    error_rate: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'inference_time_ms': self.inference_time_ms,
            'throughput_qps': self.throughput_qps,
            'error_rate': self.error_rate,
            'confidence_score': self.confidence_score
        }


@dataclass
class DriftAlert:
    """Model drift detection alert."""
    model_name: str
    drift_type: DriftType
    severity: ModelHealth
    detected_at: datetime
    metrics: Dict[str, float]
    threshold_violated: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'metrics': self.metrics,
            'threshold_violated': self.threshold_violated,
            'recommendation': self.recommendation
        }


class ModelPerformanceTracker:
    """Tracks model performance metrics over time."""
    
    def __init__(self, model_name: str, window_size: int = 1000):
        self.model_name = model_name
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(f"{__name__}.ModelPerformanceTracker")
        
        # Performance baselines (established during model validation)
        self.baselines = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1_score': 0.80,
            'inference_time_ms': 100.0,
            'error_rate': 0.05
        }
        
        # Alert thresholds
        self.warning_thresholds = {
            'accuracy_drop': 0.05,  # 5% drop from baseline
            'error_rate_increase': 0.02,  # 2% increase
            'latency_increase': 50.0,  # 50ms increase
            'throughput_drop': 0.20  # 20% drop
        }
        
        self.critical_thresholds = {
            'accuracy_drop': 0.10,  # 10% drop from baseline
            'error_rate_increase': 0.05,  # 5% increase
            'latency_increase': 200.0,  # 200ms increase
            'throughput_drop': 0.40  # 40% drop
        }
    
    def record_metrics(self, metrics: ModelMetrics):
        """Record new model metrics."""
        self.metrics_history.append(metrics)
        self.logger.debug(f"Recorded metrics for {self.model_name}: {metrics.to_dict()}")
    
    def get_recent_metrics(self, hours: int = 24) -> List[ModelMetrics]:
        """Get metrics from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def calculate_moving_average(self, metric_name: str, window: int = 10) -> Optional[float]:
        """Calculate moving average for a specific metric."""
        recent_metrics = list(self.metrics_history)[-window:]
        values = [getattr(m, metric_name) for m in recent_metrics if getattr(m, metric_name) is not None]
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    def detect_performance_drift(self) -> List[DriftAlert]:
        """Detect performance drift based on historical metrics."""
        alerts = []
        
        if len(self.metrics_history) < 20:  # Need sufficient history
            return alerts
        
        recent_avg = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time_ms', 'error_rate']:
            recent_avg[metric_name] = self.calculate_moving_average(metric_name, 10)
        
        # Check accuracy drift
        if recent_avg['accuracy'] and self.baselines['accuracy']:
            accuracy_drop = self.baselines['accuracy'] - recent_avg['accuracy']
            
            if accuracy_drop >= self.critical_thresholds['accuracy_drop']:
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=ModelHealth.CRITICAL,
                    detected_at=datetime.now(),
                    metrics={'accuracy_drop': accuracy_drop, 'current_accuracy': recent_avg['accuracy']},
                    threshold_violated=f"Accuracy dropped by {accuracy_drop:.3f} (>{self.critical_thresholds['accuracy_drop']:.3f})",
                    recommendation="Immediate model retraining or rollback recommended"
                ))
            elif accuracy_drop >= self.warning_thresholds['accuracy_drop']:
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=ModelHealth.WARNING,
                    detected_at=datetime.now(),
                    metrics={'accuracy_drop': accuracy_drop, 'current_accuracy': recent_avg['accuracy']},
                    threshold_violated=f"Accuracy dropped by {accuracy_drop:.3f} (>{self.warning_thresholds['accuracy_drop']:.3f})",
                    recommendation="Monitor closely, consider model refresh"
                ))
        
        # Check error rate drift
        if recent_avg['error_rate'] and self.baselines['error_rate']:
            error_rate_increase = recent_avg['error_rate'] - self.baselines['error_rate']
            
            if error_rate_increase >= self.critical_thresholds['error_rate_increase']:
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=ModelHealth.CRITICAL,
                    detected_at=datetime.now(),
                    metrics={'error_rate_increase': error_rate_increase, 'current_error_rate': recent_avg['error_rate']},
                    threshold_violated=f"Error rate increased by {error_rate_increase:.3f} (>{self.critical_thresholds['error_rate_increase']:.3f})",
                    recommendation="Investigate errors and consider model rollback"
                ))
        
        # Check latency drift
        if recent_avg['inference_time_ms'] and self.baselines['inference_time_ms']:
            latency_increase = recent_avg['inference_time_ms'] - self.baselines['inference_time_ms']
            
            if latency_increase >= self.critical_thresholds['latency_increase']:
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=ModelHealth.CRITICAL,
                    detected_at=datetime.now(),
                    metrics={'latency_increase': latency_increase, 'current_latency': recent_avg['inference_time_ms']},
                    threshold_violated=f"Latency increased by {latency_increase:.1f}ms (>{self.critical_thresholds['latency_increase']:.1f}ms)",
                    recommendation="Optimize model inference or scale resources"
                ))
        
        return alerts


class DataDriftDetector:
    """Detects data drift in model inputs."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.reference_stats = {}  # Statistical profile of training data
        self.current_window = deque(maxlen=1000)
        self.logger = logging.getLogger(f"{__name__}.DataDriftDetector")
    
    def set_reference_profile(self, reference_data: Dict[str, Any]):
        """Set reference data profile for drift detection."""
        self.reference_stats = reference_data
        self.logger.info(f"Set reference profile for {self.model_name}")
    
    def add_sample(self, features: Dict[str, Any]):
        """Add new data sample for drift monitoring."""
        self.current_window.append({
            'timestamp': datetime.now(),
            'features': features
        })
    
    def detect_drift(self) -> List[DriftAlert]:
        """Detect statistical drift in input data."""
        alerts = []
        
        if not self.reference_stats or len(self.current_window) < 100:
            return alerts
        
        # Calculate current statistics
        current_stats = self._calculate_stats()
        
        # Compare with reference statistics
        for feature_name, ref_stat in self.reference_stats.items():
            if feature_name not in current_stats:
                continue
                
            current_stat = current_stats[feature_name]
            
            # Statistical tests for drift detection
            drift_score = self._calculate_drift_score(ref_stat, current_stat)
            
            if drift_score > 0.8:  # High drift threshold
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=ModelHealth.CRITICAL,
                    detected_at=datetime.now(),
                    metrics={'drift_score': drift_score, 'feature': feature_name},
                    threshold_violated=f"Data drift score {drift_score:.3f} for feature {feature_name}",
                    recommendation="Data distribution has changed significantly - consider model retraining"
                ))
            elif drift_score > 0.6:  # Medium drift threshold
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=ModelHealth.WARNING,
                    detected_at=datetime.now(),
                    metrics={'drift_score': drift_score, 'feature': feature_name},
                    threshold_violated=f"Data drift score {drift_score:.3f} for feature {feature_name}",
                    recommendation="Monitor data quality and consider data pipeline updates"
                ))
        
        return alerts
    
    def _calculate_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary of current data window."""
        if not self.current_window:
            return {}
        
        stats = {}
        
        # Get all feature names
        all_features = set()
        for sample in self.current_window:
            all_features.update(sample['features'].keys())
        
        for feature_name in all_features:
            values = []
            for sample in self.current_window:
                if feature_name in sample['features']:
                    val = sample['features'][feature_name]
                    if isinstance(val, (int, float)):
                        values.append(val)
            
            if values:
                stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats
    
    def _calculate_drift_score(self, ref_stat: Dict[str, float], current_stat: Dict[str, float]) -> float:
        """Calculate drift score between reference and current statistics."""
        if not ref_stat or not current_stat:
            return 0.0
        
        # Simple statistical distance calculation
        mean_diff = abs(ref_stat.get('mean', 0) - current_stat.get('mean', 0))
        std_diff = abs(ref_stat.get('std', 1) - current_stat.get('std', 1))
        
        # Normalize by reference values
        ref_mean = abs(ref_stat.get('mean', 1))
        ref_std = ref_stat.get('std', 1)
        
        normalized_mean_diff = mean_diff / max(ref_mean, 0.001)
        normalized_std_diff = std_diff / max(ref_std, 0.001)
        
        # Combined drift score
        drift_score = min(1.0, (normalized_mean_diff + normalized_std_diff) / 2)
        
        return drift_score


class MLOpsMonitor:
    """Comprehensive MLOps monitoring system."""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]], alert_webhook_url: Optional[str] = None):
        self.models_config = models_config
        self.alert_webhook_url = alert_webhook_url
        self.logger = logging.getLogger(f"{__name__}.MLOpsMonitor")
        
        # Initialize trackers
        self.performance_trackers = {}
        self.drift_detectors = {}
        
        for model_name, config in models_config.items():
            self.performance_trackers[model_name] = ModelPerformanceTracker(model_name)
            self.drift_detectors[model_name] = DataDriftDetector(model_name)
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=10000)
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self.setup_prometheus_metrics()
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = None
    
    def setup_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.prom_metrics = {
            'model_accuracy': Gauge('model_accuracy', 'Model accuracy score', ['model_name', 'version'], registry=self.registry),
            'model_latency': Histogram('model_inference_time_seconds', 'Model inference time', ['model_name', 'version'], registry=self.registry),
            'model_errors': Counter('model_errors_total', 'Total model errors', ['model_name', 'version', 'error_type'], registry=self.registry),
            'model_predictions': Counter('model_predictions_total', 'Total model predictions', ['model_name', 'version'], registry=self.registry),
            'drift_alerts': Counter('model_drift_alerts_total', 'Total drift alerts', ['model_name', 'drift_type', 'severity'], registry=self.registry)
        }
    
    async def record_prediction(self, model_name: str, version: str, 
                               prediction_data: Dict[str, Any], 
                               performance_metrics: Optional[Dict[str, float]] = None):
        """Record a model prediction with performance metrics."""
        
        # Record performance metrics
        if performance_metrics:
            metrics = ModelMetrics(
                model_name=model_name,
                version=version,
                timestamp=datetime.now(),
                **performance_metrics
            )
            
            if model_name in self.performance_trackers:
                self.performance_trackers[model_name].record_metrics(metrics)
        
        # Record input data for drift detection
        if 'input_features' in prediction_data and model_name in self.drift_detectors:
            self.drift_detectors[model_name].add_sample(prediction_data['input_features'])
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prom_metrics'):
            self.prom_metrics['model_predictions'].labels(model_name=model_name, version=version).inc()
            
            if performance_metrics:
                if 'accuracy' in performance_metrics:
                    self.prom_metrics['model_accuracy'].labels(model_name=model_name, version=version).set(performance_metrics['accuracy'])
                
                if 'inference_time_ms' in performance_metrics:
                    self.prom_metrics['model_latency'].labels(model_name=model_name, version=version).observe(performance_metrics['inference_time_ms'] / 1000)
    
    async def run_drift_detection(self) -> List[DriftAlert]:
        """Run drift detection across all monitored models."""
        all_alerts = []
        
        for model_name in self.models_config.keys():
            if model_name in self.performance_trackers:
                performance_alerts = self.performance_trackers[model_name].detect_performance_drift()
                all_alerts.extend(performance_alerts)
            
            if model_name in self.drift_detectors:
                data_alerts = self.drift_detectors[model_name].detect_drift()
                all_alerts.extend(data_alerts)
        
        # Process new alerts
        for alert in all_alerts:
            await self.handle_alert(alert)
        
        return all_alerts
    
    async def handle_alert(self, alert: DriftAlert):
        """Handle a drift detection alert."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prom_metrics'):
            self.prom_metrics['drift_alerts'].labels(
                model_name=alert.model_name,
                drift_type=alert.drift_type.value,
                severity=alert.severity.value
            ).inc()
        
        # Log alert
        self.logger.warning(f"MLOps Alert: {alert.model_name} - {alert.drift_type.value} - {alert.severity.value}")
        self.logger.warning(f"Details: {alert.threshold_violated}")
        self.logger.warning(f"Recommendation: {alert.recommendation}")
        
        # Send webhook notification (if configured)
        if self.alert_webhook_url:
            await self.send_webhook_alert(alert)
    
    async def send_webhook_alert(self, alert: DriftAlert):
        """Send alert via webhook."""
        try:
            import aiohttp
            
            payload = {
                'type': 'mlops_alert',
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.alert_webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Alert webhook sent successfully for {alert.model_name}")
                    else:
                        self.logger.error(f"Alert webhook failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def get_model_health_status(self, model_name: str) -> ModelHealth:
        """Get current health status for a model."""
        # Check for active critical alerts
        critical_alerts = [a for a in self.active_alerts 
                          if a.model_name == model_name and a.severity == ModelHealth.CRITICAL]
        if critical_alerts:
            return ModelHealth.CRITICAL
        
        # Check for warning alerts
        warning_alerts = [a for a in self.active_alerts 
                         if a.model_name == model_name and a.severity == ModelHealth.WARNING]
        if warning_alerts:
            return ModelHealth.WARNING
        
        # Check recent performance
        if model_name in self.performance_trackers:
            tracker = self.performance_trackers[model_name]
            recent_metrics = tracker.get_recent_metrics(hours=1)
            
            if recent_metrics:
                avg_accuracy = sum(m.accuracy for m in recent_metrics if m.accuracy) / len([m for m in recent_metrics if m.accuracy])
                if avg_accuracy < tracker.baselines.get('accuracy', 0.8) * 0.9:  # 10% drop
                    return ModelHealth.DEGRADED
        
        return ModelHealth.HEALTHY
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report for all models."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'models': {},
            'summary': {
                'total_models': len(self.models_config),
                'healthy_models': 0,
                'warning_models': 0,
                'critical_models': 0,
                'degraded_models': 0,
                'active_alerts': len(self.active_alerts),
                'alerts_last_24h': len([a for a in self.alert_history if a.detected_at >= datetime.now() - timedelta(days=1)])
            }
        }
        
        for model_name in self.models_config.keys():
            health_status = self.get_model_health_status(model_name)
            
            model_report = {
                'health_status': health_status.value,
                'active_alerts': len([a for a in self.active_alerts if a.model_name == model_name]),
                'recent_metrics': {}
            }
            
            # Add performance metrics if available
            if model_name in self.performance_trackers:
                tracker = self.performance_trackers[model_name]
                recent_metrics = tracker.get_recent_metrics(hours=1)
                
                if recent_metrics:
                    latest_metrics = recent_metrics[-1]
                    model_report['recent_metrics'] = latest_metrics.to_dict()
                    
                    # Add moving averages
                    model_report['moving_averages'] = {
                        'accuracy': tracker.calculate_moving_average('accuracy'),
                        'inference_time_ms': tracker.calculate_moving_average('inference_time_ms'),
                        'error_rate': tracker.calculate_moving_average('error_rate')
                    }
            
            report['models'][model_name] = model_report
            
            # Update summary counts
            if health_status == ModelHealth.HEALTHY:
                report['summary']['healthy_models'] += 1
            elif health_status == ModelHealth.WARNING:
                report['summary']['warning_models'] += 1
            elif health_status == ModelHealth.CRITICAL:
                report['summary']['critical_models'] += 1
            elif health_status == ModelHealth.DEGRADED:
                report['summary']['degraded_models'] += 1
        
        return report
    
    async def start_monitoring(self, check_interval_seconds: int = 300):
        """Start continuous monitoring loop."""
        self.monitoring_active = True
        self.logger.info("Starting MLOps monitoring...")
        
        while self.monitoring_active:
            try:
                # Run drift detection
                alerts = await self.run_drift_detection()
                self.last_health_check = datetime.now()
                
                if alerts:
                    self.logger.info(f"Detected {len(alerts)} new alerts")
                
                # Wait for next check
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Short delay before retry
    
    def stop_monitoring(self):
        """Stop monitoring loop."""
        self.monitoring_active = False
        self.logger.info("Stopped MLOps monitoring")
    
    async def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()


# Standalone monitoring validation
async def validate_mlops_monitoring():
    """Validate MLOps monitoring system functionality."""
    print("🔍 MLOps Monitoring System Validation")
    print("=" * 50)
    
    # Mock model configuration
    models_config = {
        'idea_classifier': {
            'version': 'v1.2.0',
            'baseline_accuracy': 0.87,
            'baseline_latency_ms': 45.0
        },
        'duplicate_detector': {
            'version': 'v2.1.0',
            'baseline_accuracy': 0.92,
            'baseline_latency_ms': 120.0
        }
    }
    
    # Initialize monitoring system
    monitor = MLOpsMonitor(models_config)
    
    print("\n1. Testing Normal Operations")
    # Simulate normal model predictions
    for i in range(50):
        await monitor.record_prediction(
            model_name='idea_classifier',
            version='v1.2.0',
            prediction_data={
                'input_features': {'text_length': 100 + i, 'complexity': 0.5 + (i * 0.01)},
                'prediction': 'valid_idea',
                'confidence': 0.85 + (i * 0.002)
            },
            performance_metrics={
                'accuracy': 0.87 + np.random.normal(0, 0.02),
                'inference_time_ms': 45.0 + np.random.normal(0, 5),
                'error_rate': 0.02 + np.random.normal(0, 0.005)
            }
        )
    
    print("   ✅ Recorded 50 normal predictions")
    
    print("\n2. Testing Performance Drift Detection")
    # Simulate performance degradation
    for i in range(20):
        await monitor.record_prediction(
            model_name='idea_classifier',
            version='v1.2.0',
            prediction_data={
                'input_features': {'text_length': 200 + i, 'complexity': 0.8},
                'prediction': 'valid_idea',
                'confidence': 0.70  # Lower confidence
            },
            performance_metrics={
                'accuracy': 0.75,  # Significant drop
                'inference_time_ms': 150.0,  # Higher latency
                'error_rate': 0.08  # Higher error rate
            }
        )
    
    # Run drift detection
    alerts = await monitor.run_drift_detection()
    print(f"   🚨 Generated {len(alerts)} drift alerts")
    
    for alert in alerts:
        print(f"      - {alert.severity.value.upper()}: {alert.threshold_violated}")
    
    print("\n3. Health Status Assessment")
    health_report = monitor.generate_health_report()
    
    print(f"   Total Models: {health_report['summary']['total_models']}")
    print(f"   Healthy: {health_report['summary']['healthy_models']}")
    print(f"   Warning: {health_report['summary']['warning_models']}")
    print(f"   Critical: {health_report['summary']['critical_models']}")
    print(f"   Active Alerts: {health_report['summary']['active_alerts']}")
    
    print("\n4. Model-specific Health Details")
    for model_name, model_data in health_report['models'].items():
        print(f"   {model_name}:")
        print(f"      Status: {model_data['health_status'].upper()}")
        print(f"      Active Alerts: {model_data['active_alerts']}")
        
        if 'moving_averages' in model_data:
            avg_data = model_data['moving_averages']
            if avg_data['accuracy']:
                print(f"      Avg Accuracy: {avg_data['accuracy']:.3f}")
            if avg_data['inference_time_ms']:
                print(f"      Avg Latency: {avg_data['inference_time_ms']:.1f}ms")
    
    print("\n5. Data Drift Detection Test")
    # Test data drift detection
    drift_detector = monitor.drift_detectors['idea_classifier']
    
    # Set reference profile
    drift_detector.set_reference_profile({
        'text_length': {'mean': 150.0, 'std': 25.0, 'min': 50, 'max': 300},
        'complexity': {'mean': 0.5, 'std': 0.2, 'min': 0.1, 'max': 0.9}
    })
    
    # Simulate data drift (different distribution)
    for i in range(100):
        drift_detector.add_sample({
            'text_length': 300 + i,  # Much higher than reference
            'complexity': 0.9  # At upper limit
        })
    
    drift_alerts = drift_detector.detect_drift()
    print(f"   📊 Detected {len(drift_alerts)} data drift alerts")
    
    for alert in drift_alerts:
        print(f"      - {alert.drift_type.value}: {alert.recommendation}")
    
    # Cleanup
    await monitor.cleanup()
    
    # Validation results
    total_alerts = len(alerts) + len(drift_alerts)
    has_performance_alerts = any(alert.severity in [ModelHealth.WARNING, ModelHealth.CRITICAL] for alert in alerts)
    has_data_drift = len(drift_alerts) > 0
    
    print(f"\n✅ MLOps Monitoring Validation Complete")
    print(f"📊 Total Alerts Generated: {total_alerts}")
    print(f"⚠️  Performance Drift Detected: {'Yes' if has_performance_alerts else 'No'}")
    print(f"📈 Data Drift Detected: {'Yes' if has_data_drift else 'No'}")
    
    return total_alerts > 0 and has_performance_alerts


if __name__ == "__main__":
    # Run standalone validation
    compliance = asyncio.run(validate_mlops_monitoring())
    print(f"\n🏆 MLOps Monitoring Target: {'✅ ACHIEVED' if compliance else '❌ NOT MET'}")
    exit(0 if compliance else 1)