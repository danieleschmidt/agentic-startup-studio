"""
Adaptive Intelligence Module - Self-learning and optimization capabilities
Implements quantum-inspired algorithms for pattern recognition and adaptation.
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of patterns the AI can recognize"""
    PERFORMANCE = "performance"
    ERROR = "error"
    USER_BEHAVIOR = "user_behavior"
    RESOURCE_USAGE = "resource_usage"
    BUSINESS_METRIC = "business_metric"
    SECURITY = "security"


class AdaptationStrategy(str, Enum):
    """Available adaptation strategies"""
    GRADIENT_OPTIMIZATION = "gradient_optimization"
    QUANTUM_ANNEALING = "quantum_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE_METHOD = "ensemble_method"


@dataclass
class Pattern:
    """Detected pattern with metadata"""
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    frequency: int
    last_seen: datetime
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, data_point: Dict[str, Any]) -> None:
        """Update pattern with new data point"""
        self.data_points.append(data_point)
        self.frequency += 1
        self.last_seen = datetime.utcnow()
        
        # Update confidence based on recent data
        if len(self.data_points) > 10:
            self.data_points = self.data_points[-10:]  # Keep last 10 points
        
        # Recalculate confidence
        self.confidence = min(0.95, self.confidence + 0.01)


@dataclass
class AdaptationRule:
    """Rule for system adaptation"""
    rule_id: str
    trigger_pattern: str
    strategy: AdaptationStrategy
    parameters: Dict[str, Any]
    success_rate: float = 0.0
    applications: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def apply_success(self) -> None:
        """Record successful application"""
        self.applications += 1
        self.success_rate = (self.success_rate * (self.applications - 1) + 1.0) / self.applications
    
    def apply_failure(self) -> None:
        """Record failed application"""
        self.applications += 1
        self.success_rate = (self.success_rate * (self.applications - 1) + 0.0) / self.applications


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization algorithms"""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.temperature = 1.0
        self.cooling_rate = 0.95
    
    def quantum_annealing_optimization(
        self, 
        objective_function: callable,
        initial_solution: Dict[str, float],
        iterations: int = 100
    ) -> Dict[str, float]:
        """Quantum annealing for parameter optimization"""
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        best_score = objective_function(current_solution)
        
        for iteration in range(iterations):
            # Generate neighbor solution with quantum tunneling
            neighbor = self._generate_quantum_neighbor(current_solution)
            neighbor_score = objective_function(neighbor)
            
            # Accept or reject based on quantum probability
            delta = neighbor_score - objective_function(current_solution)
            acceptance_probability = self._quantum_acceptance_probability(delta)
            
            if np.random.random() < acceptance_probability:
                current_solution = neighbor
                
                if neighbor_score > best_score:
                    best_solution = neighbor
                    best_score = neighbor_score
            
            # Cool down temperature
            self.temperature *= self.cooling_rate
        
        return best_solution
    
    def _generate_quantum_neighbor(self, solution: Dict[str, float]) -> Dict[str, float]:
        """Generate neighbor solution using quantum superposition"""
        neighbor = {}
        for key, value in solution.items():
            # Quantum tunneling allows larger jumps
            tunneling_probability = 0.1 * math.exp(-self.temperature)
            
            if np.random.random() < tunneling_probability:
                # Quantum tunneling - large jump
                neighbor[key] = value + np.random.normal(0, 0.5)
            else:
                # Regular perturbation
                neighbor[key] = value + np.random.normal(0, 0.1 * self.temperature)
        
        return neighbor
    
    def _quantum_acceptance_probability(self, delta: float) -> float:
        """Calculate quantum-inspired acceptance probability"""
        if delta > 0:
            return 1.0
        else:
            # Quantum tunneling probability
            return math.exp(delta / self.temperature) + 0.1 * math.exp(-abs(delta))


class AdaptiveIntelligence:
    """
    Core adaptive intelligence system that learns and optimizes automatically
    """
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.01
        self.pattern_threshold = 0.7
        self.optimizer = QuantumInspiredOptimizer()
        self._learning_enabled = True
        
    async def start_learning(self) -> None:
        """Start continuous learning processes"""
        with tracer.start_as_current_span("adaptive_intelligence_start"):
            logger.info("Adaptive Intelligence starting learning processes")
            
            # Start background learning tasks
            asyncio.create_task(self._pattern_detection_loop())
            asyncio.create_task(self._adaptation_loop())
            asyncio.create_task(self._performance_analysis_loop())
            asyncio.create_task(self._rule_evolution_loop())
    
    async def ingest_data_point(
        self, 
        data_type: PatternType, 
        data: Dict[str, Any]
    ) -> None:
        """Ingest new data point for learning"""
        with tracer.start_as_current_span("ingest_data_point") as span:
            span.set_attributes({
                "data_type": data_type.value,
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
            })
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.utcnow().isoformat()
            
            # Store for pattern detection
            self.performance_history.append({
                "type": data_type.value,
                "data": data,
                "ingested_at": datetime.utcnow()
            })
            
            # Trigger immediate pattern analysis for critical data
            if data_type in [PatternType.ERROR, PatternType.SECURITY]:
                await self._analyze_critical_pattern(data_type, data)
    
    async def _pattern_detection_loop(self) -> None:
        """Continuous pattern detection"""
        while self._learning_enabled:
            try:
                await self._detect_patterns()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
                await asyncio.sleep(60)
    
    async def _detect_patterns(self) -> None:
        """Detect patterns in recent data"""
        with tracer.start_as_current_span("detect_patterns"):
            # Group data by type
            recent_data = [
                item for item in self.performance_history 
                if item["ingested_at"] > datetime.utcnow() - timedelta(minutes=30)
            ]
            
            data_by_type = {}
            for item in recent_data:
                data_type = item["type"]
                if data_type not in data_by_type:
                    data_by_type[data_type] = []
                data_by_type[data_type].append(item["data"])
            
            # Analyze each type for patterns
            for data_type, data_points in data_by_type.items():
                if len(data_points) >= 5:  # Need minimum data points
                    await self._analyze_pattern_type(PatternType(data_type), data_points)
    
    async def _analyze_pattern_type(
        self, 
        pattern_type: PatternType, 
        data_points: List[Dict[str, Any]]
    ) -> None:
        """Analyze specific pattern type"""
        with tracer.start_as_current_span("analyze_pattern_type") as span:
            span.set_attribute("pattern_type", pattern_type.value)
            
            if pattern_type == PatternType.PERFORMANCE:
                await self._analyze_performance_patterns(data_points)
            elif pattern_type == PatternType.ERROR:
                await self._analyze_error_patterns(data_points)
            elif pattern_type == PatternType.RESOURCE_USAGE:
                await self._analyze_resource_patterns(data_points)
    
    async def _analyze_performance_patterns(self, data_points: List[Dict[str, Any]]) -> None:
        """Analyze performance patterns"""
        # Extract response times
        response_times = [
            point.get("response_time", 0) for point in data_points 
            if "response_time" in point
        ]
        
        if len(response_times) >= 5:
            avg_response_time = np.mean(response_times)
            trend = self._calculate_trend(response_times)
            
            if avg_response_time > 500 and trend > 0.1:  # Degrading performance
                pattern_id = f"performance_degradation_{int(time.time())}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.PERFORMANCE,
                    confidence=0.8,
                    frequency=1,
                    last_seen=datetime.utcnow(),
                    data_points=data_points,
                    metadata={"avg_response_time": avg_response_time, "trend": trend}
                )
                
                self.patterns[pattern_id] = pattern
                await self._create_adaptation_rule(pattern)
    
    async def _analyze_error_patterns(self, data_points: List[Dict[str, Any]]) -> None:
        """Analyze error patterns"""
        error_types = {}
        for point in data_points:
            error_type = point.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Find frequent error types
        for error_type, count in error_types.items():
            if count >= 3:  # Frequent error
                pattern_id = f"error_{error_type}_{int(time.time())}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.ERROR,
                    confidence=min(0.9, count / len(data_points)),
                    frequency=count,
                    last_seen=datetime.utcnow(),
                    data_points=data_points,
                    metadata={"error_type": error_type, "frequency": count}
                )
                
                self.patterns[pattern_id] = pattern
                await self._create_adaptation_rule(pattern)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return np.sign(coeffs[0]) * min(1.0, abs(coeffs[0]) / max(values))
    
    async def _create_adaptation_rule(self, pattern: Pattern) -> None:
        """Create adaptation rule based on pattern"""
        with tracer.start_as_current_span("create_adaptation_rule"):
            rule_id = f"rule_{pattern.pattern_id}"
            
            # Determine best strategy based on pattern type
            strategy = self._select_adaptation_strategy(pattern)
            parameters = self._generate_adaptation_parameters(pattern, strategy)
            
            rule = AdaptationRule(
                rule_id=rule_id,
                trigger_pattern=pattern.pattern_id,
                strategy=strategy,
                parameters=parameters
            )
            
            self.adaptation_rules[rule_id] = rule
            logger.info(f"Created adaptation rule: {rule_id} for pattern: {pattern.pattern_id}")
    
    def _select_adaptation_strategy(self, pattern: Pattern) -> AdaptationStrategy:
        """Select best adaptation strategy for pattern"""
        if pattern.pattern_type == PatternType.PERFORMANCE:
            return AdaptationStrategy.QUANTUM_ANNEALING
        elif pattern.pattern_type == PatternType.ERROR:
            return AdaptationStrategy.REINFORCEMENT_LEARNING
        elif pattern.pattern_type == PatternType.RESOURCE_USAGE:
            return AdaptationStrategy.GRADIENT_OPTIMIZATION
        else:
            return AdaptationStrategy.ENSEMBLE_METHOD
    
    def _generate_adaptation_parameters(
        self, 
        pattern: Pattern, 
        strategy: AdaptationStrategy
    ) -> Dict[str, Any]:
        """Generate parameters for adaptation strategy"""
        base_params = {
            "pattern_confidence": pattern.confidence,
            "pattern_frequency": pattern.frequency
        }
        
        if strategy == AdaptationStrategy.QUANTUM_ANNEALING:
            return {
                **base_params,
                "temperature": 1.0,
                "cooling_rate": 0.95,
                "iterations": 100
            }
        elif strategy == AdaptationStrategy.REINFORCEMENT_LEARNING:
            return {
                **base_params,
                "learning_rate": 0.01,
                "exploration_rate": 0.1,
                "discount_factor": 0.95
            }
        else:
            return base_params
    
    async def _adaptation_loop(self) -> None:
        """Continuous adaptation based on rules"""
        while self._learning_enabled:
            try:
                await self._apply_adaptations()
                await asyncio.sleep(60)  # Apply every minute
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(120)
    
    async def _apply_adaptations(self) -> None:
        """Apply active adaptation rules"""
        with tracer.start_as_current_span("apply_adaptations"):
            active_rules = [
                rule for rule in self.adaptation_rules.values()
                if rule.success_rate >= 0.3  # Only apply successful rules
            ]
            
            for rule in active_rules:
                if rule.trigger_pattern in self.patterns:
                    pattern = self.patterns[rule.trigger_pattern]
                    if pattern.confidence >= self.pattern_threshold:
                        await self._execute_adaptation(rule, pattern)
    
    async def _execute_adaptation(self, rule: AdaptationRule, pattern: Pattern) -> None:
        """Execute specific adaptation"""
        with tracer.start_as_current_span("execute_adaptation") as span:
            span.set_attributes({
                "rule_id": rule.rule_id,
                "strategy": rule.strategy.value
            })
            
            try:
                if rule.strategy == AdaptationStrategy.QUANTUM_ANNEALING:
                    await self._apply_quantum_optimization(rule, pattern)
                elif rule.strategy == AdaptationStrategy.REINFORCEMENT_LEARNING:
                    await self._apply_reinforcement_learning(rule, pattern)
                
                rule.apply_success()
                logger.info(f"Successfully applied adaptation rule: {rule.rule_id}")
                
            except Exception as e:
                rule.apply_failure()
                logger.error(f"Failed to apply adaptation rule {rule.rule_id}: {e}")
    
    async def _apply_quantum_optimization(self, rule: AdaptationRule, pattern: Pattern) -> None:
        """Apply quantum-inspired optimization"""
        # Define objective function based on pattern
        def objective_function(params: Dict[str, float]) -> float:
            # Simulate optimization target (maximize performance, minimize errors)
            if pattern.pattern_type == PatternType.PERFORMANCE:
                return -params.get("response_time_target", 200)  # Minimize response time
            else:
                return -params.get("error_rate", 0.01)  # Minimize errors
        
        # Initial solution
        initial_solution = {
            "response_time_target": 200.0,
            "error_rate": 0.01,
            "resource_allocation": 1.0
        }
        
        # Optimize
        optimized_params = self.optimizer.quantum_annealing_optimization(
            objective_function, initial_solution
        )
        
        # Apply optimized parameters (in real system, would update configuration)
        logger.info(f"Quantum optimization result: {optimized_params}")
    
    async def _apply_reinforcement_learning(self, rule: AdaptationRule, pattern: Pattern) -> None:
        """Apply reinforcement learning adaptation"""
        # Simple Q-learning implementation
        learning_rate = rule.parameters.get("learning_rate", 0.01)
        
        # Calculate reward based on pattern improvement
        if pattern.pattern_type == PatternType.ERROR:
            # Reward for reducing error frequency
            reward = -pattern.frequency * 0.1
        else:
            reward = pattern.confidence * 0.1
        
        # Update rule success rate (simplified Q-learning)
        rule.success_rate += learning_rate * (reward - rule.success_rate)
        
        logger.info(f"RL adaptation reward: {reward}, updated success rate: {rule.success_rate}")
    
    async def _performance_analysis_loop(self) -> None:
        """Analyze overall system performance"""
        while self._learning_enabled:
            try:
                await self._analyze_system_performance()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_system_performance(self) -> None:
        """Analyze overall system performance"""
        with tracer.start_as_current_span("analyze_system_performance"):
            # Calculate performance metrics
            recent_data = [
                item for item in self.performance_history
                if item["ingested_at"] > datetime.utcnow() - timedelta(minutes=60)
            ]
            
            performance_score = self._calculate_performance_score(recent_data)
            adaptation_effectiveness = self._calculate_adaptation_effectiveness()
            
            logger.info(f"System performance score: {performance_score:.2f}")
            logger.info(f"Adaptation effectiveness: {adaptation_effectiveness:.2f}")
    
    def _calculate_performance_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score (0-1)"""
        if not data:
            return 0.5
        
        performance_points = []
        for item in data:
            if item["type"] == PatternType.PERFORMANCE.value:
                response_time = item["data"].get("response_time", 200)
                # Score based on response time (lower is better)
                score = max(0, 1 - (response_time - 100) / 900)  # 100-1000ms range
                performance_points.append(score)
        
        return np.mean(performance_points) if performance_points else 0.5
    
    def _calculate_adaptation_effectiveness(self) -> float:
        """Calculate effectiveness of adaptations"""
        if not self.adaptation_rules:
            return 0.0
        
        success_rates = [rule.success_rate for rule in self.adaptation_rules.values()]
        return np.mean(success_rates)
    
    async def _rule_evolution_loop(self) -> None:
        """Evolve adaptation rules using genetic algorithms"""
        while self._learning_enabled:
            try:
                await self._evolve_rules()
                await asyncio.sleep(600)  # Evolve every 10 minutes
            except Exception as e:
                logger.error(f"Error in rule evolution: {e}")
                await asyncio.sleep(600)
    
    async def _evolve_rules(self) -> None:
        """Evolve adaptation rules using genetic algorithms"""
        with tracer.start_as_current_span("evolve_rules"):
            if len(self.adaptation_rules) < 2:
                return
            
            # Select best performing rules
            sorted_rules = sorted(
                self.adaptation_rules.values(),
                key=lambda r: r.success_rate,
                reverse=True
            )
            
            # Keep top 50% and create new rules through crossover
            top_rules = sorted_rules[:len(sorted_rules)//2]
            
            for i in range(len(top_rules)//2):
                parent1, parent2 = top_rules[i*2:(i+1)*2]
                child_rule = self._crossover_rules(parent1, parent2)
                self.adaptation_rules[child_rule.rule_id] = child_rule
            
            logger.info(f"Rule evolution complete: {len(top_rules)} survivors, {len(top_rules)//2} new rules")
    
    def _crossover_rules(self, parent1: AdaptationRule, parent2: AdaptationRule) -> AdaptationRule:
        """Create new rule by crossing over two parent rules"""
        child_id = f"evolved_{int(time.time())}_{np.random.randint(1000)}"
        
        # Combine parameters from both parents
        child_params = {}
        for key in set(parent1.parameters.keys()) | set(parent2.parameters.keys()):
            if np.random.random() < 0.5:
                child_params[key] = parent1.parameters.get(key, parent2.parameters.get(key))
            else:
                child_params[key] = parent2.parameters.get(key, parent1.parameters.get(key))
        
        # Mutate some parameters
        for key, value in child_params.items():
            if isinstance(value, (int, float)) and np.random.random() < 0.1:  # 10% mutation rate
                child_params[key] = value * (1 + np.random.normal(0, 0.1))
        
        return AdaptationRule(
            rule_id=child_id,
            trigger_pattern=parent1.trigger_pattern if np.random.random() < 0.5 else parent2.trigger_pattern,
            strategy=parent1.strategy if parent1.success_rate > parent2.success_rate else parent2.strategy,
            parameters=child_params
        )
    
    async def _analyze_critical_pattern(self, data_type: PatternType, data: Dict[str, Any]) -> None:
        """Immediate analysis for critical patterns"""
        with tracer.start_as_current_span("analyze_critical_pattern"):
            if data_type == PatternType.ERROR:
                error_type = data.get("error_type", "unknown")
                if error_type in ["security_breach", "data_corruption", "system_failure"]:
                    # Create immediate adaptation rule
                    pattern_id = f"critical_{error_type}_{int(time.time())}"
                    pattern = Pattern(
                        pattern_id=pattern_id,
                        pattern_type=data_type,
                        confidence=0.95,
                        frequency=1,
                        last_seen=datetime.utcnow(),
                        metadata={"critical": True, "error_type": error_type}
                    )
                    
                    self.patterns[pattern_id] = pattern
                    await self._create_adaptation_rule(pattern)
                    
                    logger.warning(f"Critical pattern detected: {error_type}")
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive intelligence report"""
        with tracer.start_as_current_span("intelligence_report"):
            patterns_by_type = {}
            for pattern in self.patterns.values():
                pattern_type = pattern.pattern_type.value
                if pattern_type not in patterns_by_type:
                    patterns_by_type[pattern_type] = 0
                patterns_by_type[pattern_type] += 1
            
            adaptation_success = np.mean([
                rule.success_rate for rule in self.adaptation_rules.values()
            ]) if self.adaptation_rules else 0.0
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "patterns_detected": len(self.patterns),
                "patterns_by_type": patterns_by_type,
                "adaptation_rules": len(self.adaptation_rules),
                "adaptation_success_rate": adaptation_success,
                "data_points_processed": len(self.performance_history),
                "learning_enabled": self._learning_enabled,
                "quantum_optimizer_temperature": self.optimizer.temperature
            }


# Global intelligence instance
_intelligence: Optional[AdaptiveIntelligence] = None


async def get_intelligence() -> AdaptiveIntelligence:
    """Get or create the global adaptive intelligence instance"""
    global _intelligence
    if _intelligence is None:
        _intelligence = AdaptiveIntelligence()
        await _intelligence.start_learning()
    return _intelligence


async def ingest_performance_data(data: Dict[str, Any]) -> None:
    """Convenience function to ingest performance data"""
    intelligence = await get_intelligence()
    await intelligence.ingest_data_point(PatternType.PERFORMANCE, data)


async def ingest_error_data(data: Dict[str, Any]) -> None:
    """Convenience function to ingest error data"""
    intelligence = await get_intelligence()
    await intelligence.ingest_data_point(PatternType.ERROR, data)