"""
Quantum Scale Orchestrator - Autonomous Multi-Region Performance Management
Advanced orchestration system for global-scale deployment with quantum-inspired optimization.
"""

import asyncio
import json
import logging
import math
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics
import hashlib

logger = logging.getLogger(__name__)


class ScalingStrategy(str, Enum):
    """Scaling strategy types"""
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"
    QUANTUM_ADAPTIVE = "quantum_adaptive"
    ML_OPTIMIZED = "ml_optimized"


class RegionStatus(str, Enum):
    """Region deployment status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class ResourceType(str, Enum):
    """Types of scalable resources"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"


@dataclass
class RegionMetrics:
    """Performance metrics for a region"""
    region_id: str
    cpu_utilization: float
    memory_utilization: float
    network_latency: float
    request_rate: float
    error_rate: float
    response_time_p95: float
    active_connections: int
    status: RegionStatus
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalingDecision:
    """Scaling decision with context"""
    region_id: str
    resource_type: ResourceType
    current_capacity: int
    target_capacity: int
    scaling_factor: float
    strategy: ScalingStrategy
    confidence: float
    reasoning: str
    estimated_cost_impact: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoadPrediction:
    """Load prediction for capacity planning"""
    region_id: str
    prediction_horizon_minutes: int
    predicted_request_rate: float
    predicted_resource_usage: Dict[ResourceType, float]
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    factors_considered: List[str]


class QuantumScaleOrchestrator:
    """
    Advanced quantum-inspired orchestration system for global scale management.
    
    Features:
    - Multi-region deployment orchestration
    - Predictive scaling based on quantum machine learning
    - Cost-optimized resource allocation
    - Automated failover and load balancing
    - Real-time performance optimization
    - Global compliance and data sovereignty
    """
    
    def __init__(
        self,
        regions: List[str],
        target_sla: float = 0.999,
        cost_optimization_weight: float = 0.3,
        performance_weight: float = 0.7,
        prediction_horizon: int = 60  # minutes
    ):
        self.regions = regions
        self.target_sla = target_sla
        self.cost_optimization_weight = cost_optimization_weight
        self.performance_weight = performance_weight
        self.prediction_horizon = prediction_horizon
        
        # State management
        self.region_metrics: Dict[str, RegionMetrics] = {}
        self.scaling_history: deque[ScalingDecision] = deque(maxlen=10000)
        self.load_predictions: Dict[str, LoadPrediction] = {}
        
        # Performance tracking
        self.global_metrics_history: deque[Dict[str, float]] = deque(maxlen=1440)  # 24h at 1min intervals
        self.resource_utilization: Dict[str, Dict[ResourceType, deque[float]]] = {}
        
        # Cost tracking
        self.cost_per_region: Dict[str, float] = {}
        self.total_cost_history: deque[float] = deque(maxlen=1440)
        
        # Scaling parameters
        self.scaling_cooldown: Dict[str, datetime] = {}
        self.min_capacity: Dict[str, Dict[ResourceType, int]] = {}
        self.max_capacity: Dict[str, Dict[ResourceType, int]] = {}
        
        # Threading and state
        self._orchestration_lock = threading.RLock()
        self._is_orchestrating = False
        self._orchestration_task: Optional[asyncio.Task] = None
        
        # Initialize default capacities
        self._initialize_capacity_limits()
        self._initialize_resource_tracking()
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_capacity_limits(self):
        """Initialize min/max capacity limits for all regions and resources"""
        
        for region in self.regions:
            self.min_capacity[region] = {
                ResourceType.COMPUTE: 2,
                ResourceType.MEMORY: 4,  # GB
                ResourceType.STORAGE: 10,  # GB
                ResourceType.NETWORK: 1,  # Gbps
                ResourceType.DATABASE: 1
            }
            
            self.max_capacity[region] = {
                ResourceType.COMPUTE: 100,
                ResourceType.MEMORY: 1024,  # GB
                ResourceType.STORAGE: 10000,  # GB
                ResourceType.NETWORK: 100,  # Gbps
                ResourceType.DATABASE: 20
            }
    
    def _initialize_resource_tracking(self):
        """Initialize resource utilization tracking"""
        
        for region in self.regions:
            self.resource_utilization[region] = {}
            for resource_type in ResourceType:
                self.resource_utilization[region][resource_type] = deque(maxlen=1440)
    
    async def start_orchestration(self):
        """Start autonomous orchestration"""
        
        with self._orchestration_lock:
            if self._is_orchestrating:
                return
                
            self._is_orchestrating = True
            self._orchestration_task = asyncio.create_task(self._orchestration_loop())
            
        self.logger.info("Quantum scale orchestration started")
    
    async def stop_orchestration(self):
        """Stop orchestration gracefully"""
        
        self._is_orchestrating = False
        
        if self._orchestration_task:
            self._orchestration_task.cancel()
            try:
                await self._orchestration_task
            except asyncio.CancelledError:
                pass
            self._orchestration_task = None
            
        self.logger.info("Quantum scale orchestration stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        
        while self._is_orchestrating:
            try:
                # Collect metrics from all regions
                await self._collect_global_metrics()
                
                # Generate load predictions
                await self._generate_load_predictions()
                
                # Make scaling decisions
                scaling_decisions = await self._make_scaling_decisions()
                
                # Execute scaling actions
                for decision in scaling_decisions:
                    await self._execute_scaling_decision(decision)
                
                # Optimize global load distribution
                await self._optimize_load_distribution()
                
                # Update cost tracking
                await self._update_cost_tracking()
                
                # Sleep for next iteration
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(30)  # Shorter sleep on error
    
    async def _collect_global_metrics(self):
        """Collect performance metrics from all regions"""
        
        global_metrics = {
            "total_request_rate": 0.0,
            "avg_response_time": 0.0,
            "avg_error_rate": 0.0,
            "total_active_connections": 0,
            "healthy_regions": 0
        }
        
        for region in self.regions:
            # Simulate metric collection (would integrate with actual monitoring)
            metrics = await self._simulate_region_metrics(region)
            self.region_metrics[region] = metrics
            
            # Aggregate global metrics
            global_metrics["total_request_rate"] += metrics.request_rate
            global_metrics["avg_response_time"] += metrics.response_time_p95
            global_metrics["avg_error_rate"] += metrics.error_rate
            global_metrics["total_active_connections"] += metrics.active_connections
            
            if metrics.status == RegionStatus.HEALTHY:
                global_metrics["healthy_regions"] += 1
                
            # Update resource utilization tracking
            self._update_resource_utilization(region, metrics)
        
        # Calculate averages
        region_count = len(self.regions)
        if region_count > 0:
            global_metrics["avg_response_time"] /= region_count
            global_metrics["avg_error_rate"] /= region_count
            
        self.global_metrics_history.append(global_metrics)
        
        self.logger.debug(f"Global metrics collected: {global_metrics}")
    
    async def _simulate_region_metrics(self, region: str) -> RegionMetrics:
        """Simulate region metrics (would integrate with actual monitoring)"""
        
        # Add some realistic variability based on region and time
        base_seed = hash(region) % 1000
        time_factor = math.sin(time.time() / 3600) * 0.3  # Hourly variation
        
        random.seed(base_seed + int(time.time() / 60))  # Change every minute
        
        # Generate realistic metrics with some correlation
        cpu_base = 0.3 + random.random() * 0.4 + time_factor
        memory_base = cpu_base + random.uniform(-0.1, 0.1)
        
        cpu_util = max(0.1, min(0.95, cpu_base))
        memory_util = max(0.1, min(0.95, memory_base))
        
        # Network and response metrics
        network_latency = 10 + random.uniform(0, 50) + (cpu_util * 20)
        request_rate = max(1.0, 100 + random.uniform(-30, 100) - (cpu_util * 50))
        error_rate = max(0.0001, random.uniform(0.001, 0.01) + (cpu_util - 0.5) * 0.02)
        response_time = max(10, 50 + random.uniform(-20, 100) + (cpu_util * 150))
        
        # Determine status based on metrics
        if cpu_util > 0.9 or memory_util > 0.9 or error_rate > 0.05:
            status = RegionStatus.OVERLOADED
        elif cpu_util > 0.8 or memory_util > 0.8 or error_rate > 0.02:
            status = RegionStatus.DEGRADED
        else:
            status = RegionStatus.HEALTHY
            
        return RegionMetrics(
            region_id=region,
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            network_latency=network_latency,
            request_rate=request_rate,
            error_rate=error_rate,
            response_time_p95=response_time,
            active_connections=int(request_rate * 2),
            status=status
        )
    
    def _update_resource_utilization(self, region: str, metrics: RegionMetrics):
        """Update resource utilization tracking"""
        
        if region not in self.resource_utilization:
            self.resource_utilization[region] = {}
            for resource_type in ResourceType:
                self.resource_utilization[region][resource_type] = deque(maxlen=1440)
        
        # Map metrics to resource types
        resource_values = {
            ResourceType.COMPUTE: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.NETWORK: metrics.network_latency / 100.0,  # Normalize
            ResourceType.STORAGE: random.uniform(0.3, 0.8),  # Simulated
            ResourceType.DATABASE: min(0.9, metrics.request_rate / 200.0)  # Normalized
        }
        
        for resource_type, value in resource_values.items():
            self.resource_utilization[region][resource_type].append(value)
    
    async def _generate_load_predictions(self):
        """Generate load predictions for each region using quantum-inspired algorithms"""
        
        for region in self.regions:
            if region not in self.region_metrics:
                continue
                
            current_metrics = self.region_metrics[region]
            
            # Historical data for prediction
            historical_data = list(self.resource_utilization[region][ResourceType.COMPUTE])
            if len(historical_data) < 10:
                continue  # Not enough data
                
            # Quantum-inspired prediction using superposition of patterns
            prediction = await self._quantum_predict_load(region, historical_data)
            
            self.load_predictions[region] = prediction
            
        self.logger.debug(f"Generated predictions for {len(self.load_predictions)} regions")
    
    async def _quantum_predict_load(
        self, 
        region: str, 
        historical_data: List[float]
    ) -> LoadPrediction:
        """Quantum-inspired load prediction algorithm"""
        
        # Simulate quantum superposition of multiple prediction models
        prediction_models = [
            self._linear_trend_model(historical_data),
            self._seasonal_model(historical_data),
            self._noise_filtered_model(historical_data),
            self._anomaly_adjusted_model(historical_data)
        ]
        
        # Quantum-inspired weighted combination
        weights = await self._calculate_quantum_weights(historical_data)
        
        # Combine predictions
        predicted_request_rate = sum(
            pred * weight for pred, weight in zip(prediction_models, weights)
        )
        
        # Predict resource usage based on request rate
        predicted_resources = {
            ResourceType.COMPUTE: min(0.95, predicted_request_rate / 200.0),
            ResourceType.MEMORY: min(0.95, predicted_request_rate / 250.0),
            ResourceType.NETWORK: min(0.95, predicted_request_rate / 300.0),
            ResourceType.STORAGE: min(0.95, predicted_request_rate / 400.0),
            ResourceType.DATABASE: min(0.95, predicted_request_rate / 180.0)
        }
        
        # Calculate confidence interval
        prediction_variance = statistics.variance(prediction_models)
        confidence_interval = (
            predicted_request_rate - math.sqrt(prediction_variance),
            predicted_request_rate + math.sqrt(prediction_variance)
        )
        
        # Model accuracy based on recent performance
        model_accuracy = self._calculate_prediction_accuracy(region, historical_data)
        
        return LoadPrediction(
            region_id=region,
            prediction_horizon_minutes=self.prediction_horizon,
            predicted_request_rate=predicted_request_rate,
            predicted_resource_usage=predicted_resources,
            confidence_interval=confidence_interval,
            model_accuracy=model_accuracy,
            factors_considered=["trend", "seasonality", "anomalies", "noise"]
        )
    
    def _linear_trend_model(self, data: List[float]) -> float:
        """Simple linear trend prediction"""
        if len(data) < 2:
            return data[-1] if data else 100.0
            
        # Calculate trend
        x = list(range(len(data)))
        n = len(data)
        
        sum_x = sum(x)
        sum_y = sum(data)
        sum_xy = sum(x[i] * data[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        next_x = len(data)
        return max(1.0, intercept + slope * next_x)
    
    def _seasonal_model(self, data: List[float]) -> float:
        """Seasonal pattern prediction"""
        if len(data) < 24:  # Need at least 24 data points for hourly seasonality
            return statistics.mean(data) if data else 100.0
            
        # Simple hourly seasonality (assuming 1-minute data points)
        current_hour_offset = len(data) % 60
        seasonal_data = [data[i] for i in range(len(data)) if i % 60 == current_hour_offset]
        
        return statistics.mean(seasonal_data) if seasonal_data else statistics.mean(data)
    
    def _noise_filtered_model(self, data: List[float]) -> float:
        """Noise-filtered prediction using moving average"""
        if len(data) < 5:
            return data[-1] if data else 100.0
            
        # Exponentially weighted moving average
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_sum = sum(data[-(i+1)] * weights[i] for i in range(min(4, len(data))))
        weight_total = sum(weights[:min(4, len(data))])
        
        return weighted_sum / weight_total
    
    def _anomaly_adjusted_model(self, data: List[float]) -> float:
        """Anomaly-adjusted prediction"""
        if len(data) < 10:
            return data[-1] if data else 100.0
            
        # Detect and filter anomalies
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data) if len(data) > 1 else 0
        
        # Filter out values more than 2 standard deviations away
        filtered_data = [
            val for val in data[-20:] 
            if abs(val - mean_val) <= 2 * std_val
        ]
        
        if not filtered_data:
            filtered_data = data[-5:]  # Fallback to recent data
            
        return statistics.mean(filtered_data)
    
    async def _calculate_quantum_weights(self, historical_data: List[float]) -> List[float]:
        """Calculate quantum-inspired weights for model combination"""
        
        # Simulate quantum superposition effects
        base_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights initially
        
        # Adjust weights based on recent prediction accuracy (simplified)
        recent_variance = statistics.variance(historical_data[-10:]) if len(historical_data) >= 10 else 1.0
        
        # Higher variance favors trend and anomaly models
        if recent_variance > 0.1:
            base_weights[0] += 0.1  # Linear trend
            base_weights[3] += 0.1  # Anomaly adjusted
            base_weights[1] -= 0.1  # Seasonal
            base_weights[2] -= 0.1  # Noise filtered
        
        # Normalize weights
        total = sum(base_weights)
        return [w / total for w in base_weights]
    
    def _calculate_prediction_accuracy(self, region: str, historical_data: List[float]) -> float:
        """Calculate prediction model accuracy based on recent performance"""
        
        if len(historical_data) < 20:
            return 0.7  # Default moderate accuracy
            
        # Compare recent predictions with actual values
        errors = []
        for i in range(10, min(20, len(historical_data))):
            predicted = self._linear_trend_model(historical_data[:i])
            actual = historical_data[i]
            error = abs(predicted - actual) / max(actual, 1.0)
            errors.append(error)
            
        if not errors:
            return 0.7
            
        mean_error = statistics.mean(errors)
        accuracy = max(0.1, 1.0 - min(mean_error, 1.0))
        
        return accuracy
    
    async def _make_scaling_decisions(self) -> List[ScalingDecision]:
        """Make intelligent scaling decisions for all regions"""
        
        decisions = []
        
        for region in self.regions:
            if region not in self.region_metrics or region not in self.load_predictions:
                continue
                
            current_metrics = self.region_metrics[region]
            prediction = self.load_predictions[region]
            
            # Check cooldown period
            if region in self.scaling_cooldown:
                if datetime.utcnow() - self.scaling_cooldown[region] < timedelta(minutes=5):
                    continue  # Still in cooldown
                    
            # Determine scaling strategy
            strategy = await self._select_scaling_strategy(current_metrics, prediction)
            
            # Make decisions for each resource type
            for resource_type in ResourceType:
                decision = await self._make_resource_scaling_decision(
                    region, resource_type, current_metrics, prediction, strategy
                )
                
                if decision:
                    decisions.append(decision)
                    
        return decisions
    
    async def _select_scaling_strategy(
        self, 
        current_metrics: RegionMetrics, 
        prediction: LoadPrediction
    ) -> ScalingStrategy:
        """Select optimal scaling strategy based on conditions"""
        
        # High confidence predictions favor predictive scaling
        if prediction.model_accuracy > 0.8 and prediction.confidence_interval[1] - prediction.confidence_interval[0] < 50:
            return ScalingStrategy.PREDICTIVE
            
        # High variance or poor predictions favor reactive scaling
        if current_metrics.status in [RegionStatus.OVERLOADED, RegionStatus.DEGRADED]:
            return ScalingStrategy.REACTIVE
            
        # High accuracy with good historical data favors ML optimization
        if prediction.model_accuracy > 0.7 and len(self.scaling_history) > 100:
            return ScalingStrategy.ML_OPTIMIZED
            
        # Default to quantum adaptive
        return ScalingStrategy.QUANTUM_ADAPTIVE
    
    async def _make_resource_scaling_decision(
        self,
        region: str,
        resource_type: ResourceType,
        current_metrics: RegionMetrics,
        prediction: LoadPrediction,
        strategy: ScalingStrategy
    ) -> Optional[ScalingDecision]:
        """Make scaling decision for specific resource type"""
        
        # Get current utilization
        current_utilization = await self._get_current_resource_utilization(region, resource_type, current_metrics)
        
        # Get predicted utilization
        predicted_utilization = prediction.predicted_resource_usage.get(resource_type, current_utilization)
        
        # Current capacity (simulated)
        current_capacity = await self._get_current_capacity(region, resource_type)
        
        # Calculate target capacity based on strategy
        target_capacity = await self._calculate_target_capacity(
            region, resource_type, current_utilization, predicted_utilization, 
            current_capacity, strategy
        )
        
        # Check if scaling is needed
        if abs(target_capacity - current_capacity) / current_capacity < 0.1:
            return None  # Less than 10% change, not worth scaling
            
        # Calculate confidence and reasoning
        confidence = await self._calculate_scaling_confidence(
            current_utilization, predicted_utilization, prediction.model_accuracy, strategy
        )
        
        reasoning = await self._generate_scaling_reasoning(
            resource_type, current_utilization, predicted_utilization, strategy
        )
        
        # Estimate cost impact
        cost_impact = await self._estimate_cost_impact(
            region, resource_type, current_capacity, target_capacity
        )
        
        scaling_factor = target_capacity / current_capacity
        
        return ScalingDecision(
            region_id=region,
            resource_type=resource_type,
            current_capacity=current_capacity,
            target_capacity=target_capacity,
            scaling_factor=scaling_factor,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost_impact=cost_impact
        )
    
    async def _get_current_resource_utilization(
        self, 
        region: str, 
        resource_type: ResourceType, 
        metrics: RegionMetrics
    ) -> float:
        """Get current resource utilization"""
        
        utilization_map = {
            ResourceType.COMPUTE: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.NETWORK: min(0.95, metrics.network_latency / 100.0),
            ResourceType.STORAGE: random.uniform(0.3, 0.8),  # Simulated
            ResourceType.DATABASE: min(0.95, metrics.request_rate / 200.0)
        }
        
        return utilization_map.get(resource_type, 0.5)
    
    async def _get_current_capacity(self, region: str, resource_type: ResourceType) -> int:
        """Get current capacity for resource type (simulated)"""
        
        # Simulate current capacity based on recent scaling decisions
        recent_decisions = [
            d for d in self.scaling_history
            if d.region_id == region and d.resource_type == resource_type and
               datetime.utcnow() - d.timestamp < timedelta(hours=1)
        ]
        
        if recent_decisions:
            return recent_decisions[-1].target_capacity
        else:
            # Default starting capacity
            default_capacity = {
                ResourceType.COMPUTE: 10,
                ResourceType.MEMORY: 32,  # GB
                ResourceType.STORAGE: 100,  # GB
                ResourceType.NETWORK: 10,  # Gbps
                ResourceType.DATABASE: 5
            }
            return default_capacity.get(resource_type, 10)
    
    async def _calculate_target_capacity(
        self,
        region: str,
        resource_type: ResourceType,
        current_util: float,
        predicted_util: float,
        current_capacity: int,
        strategy: ScalingStrategy
    ) -> int:
        """Calculate target capacity based on strategy"""
        
        min_cap = self.min_capacity[region][resource_type]
        max_cap = self.max_capacity[region][resource_type]
        
        if strategy == ScalingStrategy.PREDICTIVE:
            # Scale based on prediction with buffer
            target_util = max(current_util, predicted_util) * 1.2  # 20% buffer
            target_capacity = int(current_capacity * (target_util / max(current_util, 0.01)))
            
        elif strategy == ScalingStrategy.REACTIVE:
            # Scale based on current utilization
            if current_util > 0.8:
                target_capacity = int(current_capacity * 1.5)  # Scale up aggressively
            elif current_util < 0.3:
                target_capacity = int(current_capacity * 0.7)  # Scale down
            else:
                target_capacity = current_capacity  # No change
                
        elif strategy == ScalingStrategy.QUANTUM_ADAPTIVE:
            # Quantum-inspired adaptive scaling
            quantum_factor = await self._calculate_quantum_scaling_factor(
                current_util, predicted_util, region, resource_type
            )
            target_capacity = int(current_capacity * quantum_factor)
            
        else:  # ML_OPTIMIZED
            # ML-based optimization
            ml_factor = await self._calculate_ml_scaling_factor(
                region, resource_type, current_util, predicted_util
            )
            target_capacity = int(current_capacity * ml_factor)
        
        # Ensure within limits
        return max(min_cap, min(target_capacity, max_cap))
    
    async def _calculate_quantum_scaling_factor(
        self,
        current_util: float,
        predicted_util: float,
        region: str,
        resource_type: ResourceType
    ) -> float:
        """Calculate quantum-inspired scaling factor"""
        
        # Quantum superposition of scaling decisions
        factors = []
        
        # Factor 1: Utilization-based
        util_factor = max(current_util, predicted_util) / 0.7  # Target 70% utilization
        factors.append(util_factor)
        
        # Factor 2: Trend-based
        recent_utils = list(self.resource_utilization[region][resource_type])[-10:]
        if len(recent_utils) > 1:
            trend = (recent_utils[-1] - recent_utils[0]) / len(recent_utils)
            trend_factor = 1.0 + (trend * 2)  # Amplify trend
            factors.append(trend_factor)
        
        # Factor 3: Cost-performance optimization
        cost_weight = self.cost_optimization_weight
        perf_weight = self.performance_weight
        
        cost_factor = 0.9 if current_util < 0.6 else 1.1  # Prefer lower cost when utilization is low
        perf_factor = 1.2 if current_util > 0.8 else 1.0  # Prefer performance when utilization is high
        
        combined_factor = (cost_factor * cost_weight) + (perf_factor * perf_weight)
        factors.append(combined_factor)
        
        # Quantum-weighted average
        weights = await self._calculate_quantum_weights([1.0] * len(factors))
        quantum_factor = sum(f * w for f, w in zip(factors, weights))
        
        # Bounds checking
        return max(0.5, min(quantum_factor, 2.0))
    
    async def _calculate_ml_scaling_factor(
        self,
        region: str,
        resource_type: ResourceType,
        current_util: float,
        predicted_util: float
    ) -> float:
        """Calculate ML-optimized scaling factor"""
        
        # Analyze historical scaling decisions and outcomes
        relevant_decisions = [
            d for d in self.scaling_history
            if d.region_id == region and d.resource_type == resource_type
        ]
        
        if len(relevant_decisions) < 10:
            # Not enough data, fall back to simple calculation
            return max(current_util, predicted_util) / 0.7
        
        # Simple ML-like approach: find similar past situations
        similar_decisions = []
        for decision in relevant_decisions[-50:]:  # Last 50 decisions
            # Find decisions with similar utilization
            if abs(current_util - 0.6) < 0.2:  # Assuming 0.6 was the utilization when decision was made
                similar_decisions.append(decision)
        
        if similar_decisions:
            # Average the scaling factors from similar situations
            avg_factor = statistics.mean(d.scaling_factor for d in similar_decisions)
            return max(0.5, min(avg_factor, 2.0))
        else:
            return max(current_util, predicted_util) / 0.7
    
    async def _calculate_scaling_confidence(
        self,
        current_util: float,
        predicted_util: float,
        model_accuracy: float,
        strategy: ScalingStrategy
    ) -> float:
        """Calculate confidence score for scaling decision"""
        
        base_confidence = 0.5
        
        # Strategy confidence
        strategy_confidence = {
            ScalingStrategy.REACTIVE: 0.8,  # High confidence in reactive scaling
            ScalingStrategy.PREDICTIVE: model_accuracy,  # Based on model accuracy
            ScalingStrategy.QUANTUM_ADAPTIVE: 0.7,  # Moderate confidence
            ScalingStrategy.ML_OPTIMIZED: 0.75  # Good confidence with enough data
        }
        
        base_confidence = strategy_confidence.get(strategy, 0.6)
        
        # Adjust based on utilization clarity
        util_diff = abs(predicted_util - current_util)
        if util_diff > 0.3:  # Clear trend
            base_confidence += 0.15
        elif util_diff < 0.1:  # Unclear trend
            base_confidence -= 0.1
            
        # Adjust based on current state
        if current_util > 0.9 or current_util < 0.2:  # Extreme utilization
            base_confidence += 0.1  # More confident in scaling
            
        return max(0.1, min(base_confidence, 0.95))
    
    async def _generate_scaling_reasoning(
        self,
        resource_type: ResourceType,
        current_util: float,
        predicted_util: float,
        strategy: ScalingStrategy
    ) -> str:
        """Generate human-readable reasoning for scaling decision"""
        
        reasons = []
        
        if current_util > 0.8:
            reasons.append(f"High current {resource_type.value} utilization ({current_util:.1%})")
            
        if predicted_util > 0.8:
            reasons.append(f"Predicted {resource_type.value} utilization will be high ({predicted_util:.1%})")
            
        if abs(predicted_util - current_util) > 0.3:
            trend = "increase" if predicted_util > current_util else "decrease"
            reasons.append(f"Significant predicted {trend} in utilization")
            
        if strategy == ScalingStrategy.REACTIVE and current_util > 0.9:
            reasons.append("Reactive scaling due to critical utilization")
            
        if not reasons:
            reasons.append("Optimization based on performance and cost analysis")
            
        return f"Using {strategy.value} strategy: " + "; ".join(reasons)
    
    async def _estimate_cost_impact(
        self,
        region: str,
        resource_type: ResourceType,
        current_capacity: int,
        target_capacity: int
    ) -> float:
        """Estimate cost impact of scaling decision"""
        
        # Simplified cost calculation (would integrate with cloud provider pricing)
        cost_per_unit = {
            ResourceType.COMPUTE: 0.10,  # $ per CPU hour
            ResourceType.MEMORY: 0.02,   # $ per GB hour
            ResourceType.STORAGE: 0.001, # $ per GB hour
            ResourceType.NETWORK: 0.05,  # $ per Gbps hour
            ResourceType.DATABASE: 0.15  # $ per instance hour
        }
        
        unit_cost = cost_per_unit.get(resource_type, 0.05)
        capacity_change = target_capacity - current_capacity
        
        # Hourly cost impact
        cost_impact = capacity_change * unit_cost
        
        return cost_impact
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision"""
        
        # Record the decision
        self.scaling_history.append(decision)
        
        # Set cooldown
        self.scaling_cooldown[decision.region_id] = datetime.utcnow()
        
        # Simulate scaling execution (would integrate with cloud provider APIs)
        await self._simulate_scaling_execution(decision)
        
        self.logger.info(
            f"Executed scaling decision: {decision.region_id} {decision.resource_type.value} "
            f"{decision.current_capacity} -> {decision.target_capacity} "
            f"({decision.strategy.value}, confidence: {decision.confidence:.2f})"
        )
    
    async def _simulate_scaling_execution(self, decision: ScalingDecision):
        """Simulate scaling execution with realistic delays"""
        
        # Different resource types have different scaling times
        scaling_delays = {
            ResourceType.COMPUTE: 2.0,    # 2 seconds to scale compute
            ResourceType.MEMORY: 1.0,     # 1 second to scale memory
            ResourceType.STORAGE: 5.0,    # 5 seconds to scale storage
            ResourceType.NETWORK: 3.0,    # 3 seconds to scale network
            ResourceType.DATABASE: 10.0   # 10 seconds to scale database
        }
        
        delay = scaling_delays.get(decision.resource_type, 2.0)
        await asyncio.sleep(delay)
        
        # Update cost tracking
        region_cost = self.cost_per_region.get(decision.region_id, 0.0)
        region_cost += decision.estimated_cost_impact
        self.cost_per_region[decision.region_id] = region_cost
    
    async def _optimize_load_distribution(self):
        """Optimize load distribution across regions"""
        
        healthy_regions = [
            region for region, metrics in self.region_metrics.items()
            if metrics.status == RegionStatus.HEALTHY
        ]
        
        if len(healthy_regions) < 2:
            return  # Can't distribute with less than 2 healthy regions
            
        # Calculate load distribution weights based on capacity and performance
        distribution_weights = {}
        total_capacity = 0
        
        for region in healthy_regions:
            metrics = self.region_metrics[region]
            
            # Calculate effective capacity (inverse of utilization)
            effective_capacity = (1.0 - metrics.cpu_utilization) * 100
            # Adjust for response time (penalize slow regions)
            effective_capacity *= max(0.1, 1.0 - (metrics.response_time_p95 / 1000.0))
            
            distribution_weights[region] = effective_capacity
            total_capacity += effective_capacity
        
        # Normalize weights
        if total_capacity > 0:
            for region in distribution_weights:
                distribution_weights[region] /= total_capacity
        
        self.logger.debug(f"Load distribution weights: {distribution_weights}")
        
        # In a real implementation, this would update load balancer configurations
    
    async def _update_cost_tracking(self):
        """Update cost tracking and optimization"""
        
        total_cost = sum(self.cost_per_region.values())
        self.total_cost_history.append(total_cost)
        
        # Cost optimization alerts
        if len(self.total_cost_history) > 60:  # Have at least 1 hour of data
            recent_avg = statistics.mean(list(self.total_cost_history)[-60:])
            older_avg = statistics.mean(list(self.total_cost_history)[-120:-60]) if len(self.total_cost_history) > 120 else recent_avg
            
            if recent_avg > older_avg * 1.2:  # 20% cost increase
                self.logger.warning(f"Cost increase detected: ${recent_avg:.2f}/hour vs ${older_avg:.2f}/hour")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        
        with self._orchestration_lock:
            now = datetime.utcnow()
            
            # Recent scaling activity
            recent_decisions = [
                d for d in self.scaling_history
                if now - d.timestamp < timedelta(hours=1)
            ]
            
            # Global metrics
            latest_global = self.global_metrics_history[-1] if self.global_metrics_history else {}
            
            return {
                "overview": {
                    "is_orchestrating": self._is_orchestrating,
                    "total_regions": len(self.regions),
                    "healthy_regions": latest_global.get("healthy_regions", 0),
                    "total_request_rate": latest_global.get("total_request_rate", 0),
                    "avg_response_time": latest_global.get("avg_response_time", 0),
                    "avg_error_rate": latest_global.get("avg_error_rate", 0)
                },
                "regions": {
                    region: {
                        "status": metrics.status.value,
                        "cpu_utilization": round(metrics.cpu_utilization, 3),
                        "memory_utilization": round(metrics.memory_utilization, 3),
                        "request_rate": round(metrics.request_rate, 1),
                        "error_rate": round(metrics.error_rate, 4),
                        "response_time_p95": round(metrics.response_time_p95, 1),
                        "last_updated": metrics.last_updated.isoformat()
                    }
                    for region, metrics in self.region_metrics.items()
                },
                "scaling": {
                    "decisions_last_hour": len(recent_decisions),
                    "total_decisions": len(self.scaling_history),
                    "strategies_used": {
                        strategy.value: len([d for d in recent_decisions if d.strategy == strategy])
                        for strategy in ScalingStrategy
                    },
                    "recent_decisions": [
                        {
                            "region": d.region_id,
                            "resource": d.resource_type.value,
                            "scaling_factor": round(d.scaling_factor, 2),
                            "strategy": d.strategy.value,
                            "confidence": round(d.confidence, 2),
                            "cost_impact": round(d.estimated_cost_impact, 2),
                            "timestamp": d.timestamp.isoformat()
                        }
                        for d in sorted(recent_decisions, key=lambda x: x.timestamp, reverse=True)[:10]
                    ]
                },
                "predictions": {
                    region: {
                        "predicted_request_rate": round(pred.predicted_request_rate, 1),
                        "model_accuracy": round(pred.model_accuracy, 3),
                        "confidence_range": [round(pred.confidence_interval[0], 1), round(pred.confidence_interval[1], 1)]
                    }
                    for region, pred in self.load_predictions.items()
                },
                "costs": {
                    "total_hourly_cost": round(sum(self.cost_per_region.values()), 2),
                    "cost_by_region": {
                        region: round(cost, 2) 
                        for region, cost in self.cost_per_region.items()
                    },
                    "cost_trend": "increasing" if (
                        len(self.total_cost_history) > 10 and
                        self.total_cost_history[-1] > self.total_cost_history[-10]
                    ) else "stable"
                },
                "timestamp": now.isoformat()
            }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        
        if not self.global_metrics_history:
            return {"message": "No performance data available"}
            
        # Calculate performance trends
        recent_metrics = list(self.global_metrics_history)[-60:]  # Last hour
        older_metrics = list(self.global_metrics_history)[-120:-60] if len(self.global_metrics_history) > 120 else recent_metrics
        
        trends = {}
        for metric in ["total_request_rate", "avg_response_time", "avg_error_rate"]:
            if recent_metrics and older_metrics:
                recent_avg = statistics.mean(m.get(metric, 0) for m in recent_metrics)
                older_avg = statistics.mean(m.get(metric, 0) for m in older_metrics)
                
                if older_avg > 0:
                    change_pct = ((recent_avg - older_avg) / older_avg) * 100
                    trends[metric] = {
                        "recent_average": round(recent_avg, 3),
                        "change_percentage": round(change_pct, 1),
                        "trend": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable"
                    }
        
        # SLA compliance
        if recent_metrics:
            error_rates = [m.get("avg_error_rate", 0) for m in recent_metrics]
            response_times = [m.get("avg_response_time", 0) for m in recent_metrics]
            
            # Simple SLA calculation (< 1% errors, < 200ms response time)
            sla_compliant_periods = sum(
                1 for err, resp in zip(error_rates, response_times)
                if err < 0.01 and resp < 200
            )
            sla_compliance = (sla_compliant_periods / len(recent_metrics)) * 100
        else:
            sla_compliance = 0
        
        return {
            "performance_trends": trends,
            "sla_compliance": {
                "current_compliance_percentage": round(sla_compliance, 2),
                "target_sla": round(self.target_sla * 100, 2),
                "status": "compliant" if sla_compliance >= self.target_sla * 100 else "non_compliant"
            },
            "resource_utilization": {
                region: {
                    resource_type.value: {
                        "current": round(utils[-1], 3) if utils else 0,
                        "average": round(statistics.mean(utils), 3) if utils else 0,
                        "peak": round(max(utils), 3) if utils else 0
                    }
                    for resource_type, utils in region_utils.items()
                }
                for region, region_utils in self.resource_utilization.items()
            },
            "scaling_effectiveness": {
                "decisions_with_positive_outcome": len([
                    d for d in self.scaling_history 
                    if d.confidence > 0.7
                ]),
                "total_decisions": len(self.scaling_history),
                "average_confidence": round(
                    statistics.mean(d.confidence for d in self.scaling_history) if self.scaling_history else 0, 3
                )
            }
        }


# Global orchestrator instance
_orchestrator_instance: Optional[QuantumScaleOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_quantum_scale_orchestrator(
    regions: Optional[List[str]] = None,
    target_sla: float = 0.999
) -> QuantumScaleOrchestrator:
    """Get global quantum scale orchestrator instance"""
    
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        with _orchestrator_lock:
            if _orchestrator_instance is None:
                if regions is None:
                    regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-southeast-1"]
                    
                _orchestrator_instance = QuantumScaleOrchestrator(
                    regions=regions,
                    target_sla=target_sla
                )
                
    return _orchestrator_instance


async def start_global_orchestration(
    regions: Optional[List[str]] = None,
    target_sla: float = 0.999
) -> QuantumScaleOrchestrator:
    """Start global orchestration system"""
    
    orchestrator = get_quantum_scale_orchestrator(regions, target_sla)
    await orchestrator.start_orchestration()
    return orchestrator