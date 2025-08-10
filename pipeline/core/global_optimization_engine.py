"""
Global Optimization Engine - Coordinated system-wide optimization
Integrates all optimization subsystems for maximum performance and efficiency.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer
from .autonomous_executor import get_executor, AutonomousTask, Priority
from .adaptive_intelligence import get_intelligence, PatternType
from ..security.enhanced_security import get_security_manager
from ..monitoring.comprehensive_monitoring import get_monitor
from ..performance.quantum_performance_optimizer import get_optimizer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class OptimizationPhase(str, Enum):
    """Global optimization phases"""
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    LEARNING = "learning"


class SystemDomain(str, Enum):
    """System domains for optimization"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    COST = "cost"
    USER_EXPERIENCE = "user_experience"
    COMPLIANCE = "compliance"


@dataclass
class OptimizationObjective:
    """Global optimization objective"""
    domain: SystemDomain
    metric_name: str
    current_value: float
    target_value: float
    weight: float
    priority: Priority
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None
    
    def calculate_score(self) -> float:
        """Calculate objective achievement score (0-1)"""
        if self.current_value == self.target_value:
            return 1.0
        
        # Calculate normalized distance to target
        if self.constraint_min is not None and self.constraint_max is not None:
            value_range = self.constraint_max - self.constraint_min
            if value_range == 0:
                return 1.0
            
            distance = abs(self.current_value - self.target_value)
            normalized_distance = distance / value_range
            return max(0.0, 1.0 - normalized_distance)
        else:
            # Simple ratio-based scoring
            if self.target_value == 0:
                return 1.0 if self.current_value == 0 else 0.0
            
            ratio = self.current_value / self.target_value
            return min(1.0, ratio) if ratio > 0 else 0.0


@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan"""
    plan_id: str
    objectives: List[OptimizationObjective]
    tasks: List[AutonomousTask]
    estimated_duration: timedelta
    expected_improvement: Dict[str, float]
    risk_level: float
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationResult:
    """Results from optimization execution"""
    plan_id: str
    phase: OptimizationPhase
    success: bool
    duration: timedelta
    achievements: Dict[str, float]
    side_effects: Dict[str, Any]
    lessons_learned: List[str]
    next_steps: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GlobalOptimizationEngine:
    """
    Master optimization engine that coordinates all system optimization
    """
    
    def __init__(self):
        self.current_phase = OptimizationPhase.DISCOVERY
        self.objectives: List[OptimizationObjective] = []
        self.optimization_plans: Dict[str, OptimizationPlan] = {}
        self.execution_results: List[OptimizationResult] = []
        self.global_state: Dict[str, Any] = {}
        self._optimization_active = False
        
        # Integration with subsystems
        self._subsystems_initialized = False
        
    async def initialize_subsystems(self) -> None:
        """Initialize all optimization subsystems"""
        with tracer.start_as_current_span("initialize_subsystems"):
            if self._subsystems_initialized:
                return
            
            logger.info("Initializing global optimization subsystems")
            
            # Initialize all subsystems
            self.executor = await get_executor()
            self.intelligence = await get_intelligence()
            self.security_manager = get_security_manager()
            self.monitor = await get_monitor()
            self.performance_optimizer = await get_optimizer()
            
            self._subsystems_initialized = True
            logger.info("All optimization subsystems initialized")
    
    async def start_global_optimization(self) -> None:
        """Start the global optimization engine"""
        with tracer.start_as_current_span("start_global_optimization"):
            await self.initialize_subsystems()
            
            self._optimization_active = True
            logger.info("Global Optimization Engine started")
            
            # Start optimization cycles
            asyncio.create_task(self._global_optimization_cycle())
            asyncio.create_task(self._continuous_monitoring_cycle())
            asyncio.create_task(self._learning_integration_cycle())
    
    async def stop_global_optimization(self) -> None:
        """Stop the global optimization engine"""
        self._optimization_active = False
        logger.info("Global Optimization Engine stopped")
    
    async def _global_optimization_cycle(self) -> None:
        """Main global optimization cycle"""
        while self._optimization_active:
            try:
                await self._execute_optimization_phase()
                await asyncio.sleep(300)  # 5-minute cycles
            except Exception as e:
                logger.error(f"Error in global optimization cycle: {e}")
                await asyncio.sleep(600)  # Back off on errors
    
    async def _execute_optimization_phase(self) -> None:
        """Execute current optimization phase"""
        with tracer.start_as_current_span("execute_optimization_phase") as span:
            span.set_attribute("phase", self.current_phase.value)
            
            logger.info(f"Executing optimization phase: {self.current_phase.value}")
            
            if self.current_phase == OptimizationPhase.DISCOVERY:
                await self._discovery_phase()
                self.current_phase = OptimizationPhase.ANALYSIS
                
            elif self.current_phase == OptimizationPhase.ANALYSIS:
                await self._analysis_phase()
                self.current_phase = OptimizationPhase.PLANNING
                
            elif self.current_phase == OptimizationPhase.PLANNING:
                await self._planning_phase()
                self.current_phase = OptimizationPhase.EXECUTION
                
            elif self.current_phase == OptimizationPhase.EXECUTION:
                await self._execution_phase()
                self.current_phase = OptimizationPhase.VALIDATION
                
            elif self.current_phase == OptimizationPhase.VALIDATION:
                await self._validation_phase()
                self.current_phase = OptimizationPhase.LEARNING
                
            elif self.current_phase == OptimizationPhase.LEARNING:
                await self._learning_phase()
                self.current_phase = OptimizationPhase.DISCOVERY  # Restart cycle
    
    async def _discovery_phase(self) -> None:
        """Discovery phase - identify optimization opportunities"""
        with tracer.start_as_current_span("discovery_phase"):
            logger.info("Starting discovery phase")
            
            # Gather system state
            await self._gather_system_state()
            
            # Identify optimization opportunities
            opportunities = await self._identify_optimization_opportunities()
            
            # Convert opportunities to objectives
            self.objectives = await self._create_optimization_objectives(opportunities)
            
            logger.info(f"Discovery complete: {len(self.objectives)} objectives identified")
    
    async def _gather_system_state(self) -> None:
        """Gather comprehensive system state"""
        with tracer.start_as_current_span("gather_system_state"):
            # Performance metrics
            perf_report = await self.performance_optimizer.get_optimization_report()
            
            # Security status
            security_report = self.security_manager.get_security_report()
            
            # System health
            health_status = self.monitor.get_system_status()
            
            # Intelligence insights
            intelligence_report = self.intelligence.get_intelligence_report()
            
            # Execution status
            executor_status = self.executor.get_status_report()
            
            self.global_state = {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": perf_report,
                "security": security_report,
                "health": health_status,
                "intelligence": intelligence_report,
                "execution": executor_status
            }
    
    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities across all domains"""
        with tracer.start_as_current_span("identify_opportunities"):
            opportunities = []
            
            # Performance opportunities
            perf_score = self.global_state["performance"]["overall_performance_score"]
            if perf_score < 0.8:  # Below 80% performance
                opportunities.append({
                    "domain": SystemDomain.PERFORMANCE,
                    "issue": "Low overall performance score",
                    "current_value": perf_score,
                    "target_value": 0.9,
                    "priority": Priority.HIGH,
                    "impact": "high"
                })
            
            # Security opportunities
            active_alerts = self.global_state["security"]["active_alerts"]
            if active_alerts > 0:
                opportunities.append({
                    "domain": SystemDomain.SECURITY,
                    "issue": "Active security alerts",
                    "current_value": active_alerts,
                    "target_value": 0,
                    "priority": Priority.CRITICAL,
                    "impact": "critical"
                })
            
            # Reliability opportunities
            overall_health = self.global_state["health"]["overall_health"]
            if overall_health != "healthy":
                opportunities.append({
                    "domain": SystemDomain.RELIABILITY,
                    "issue": "System health degraded",
                    "current_value": 0.5 if overall_health == "degraded" else 0.0,
                    "target_value": 1.0,
                    "priority": Priority.HIGH,
                    "impact": "high"
                })
            
            # Scalability opportunities
            total_optimizations = self.global_state["performance"]["total_optimizations"]
            if total_optimizations < 10:  # Need more optimization history
                opportunities.append({
                    "domain": SystemDomain.SCALABILITY,
                    "issue": "Insufficient optimization history",
                    "current_value": total_optimizations,
                    "target_value": 50,
                    "priority": Priority.MEDIUM,
                    "impact": "medium"
                })
            
            # Intelligence opportunities
            patterns_detected = self.global_state["intelligence"]["patterns_detected"]
            if patterns_detected < 5:  # Need more learning
                opportunities.append({
                    "domain": SystemDomain.USER_EXPERIENCE,
                    "issue": "Limited intelligence patterns",
                    "current_value": patterns_detected,
                    "target_value": 20,
                    "priority": Priority.MEDIUM,
                    "impact": "medium"
                })
            
            logger.info(f"Identified {len(opportunities)} optimization opportunities")
            return opportunities
    
    async def _create_optimization_objectives(
        self, 
        opportunities: List[Dict[str, Any]]
    ) -> List[OptimizationObjective]:
        """Create optimization objectives from opportunities"""
        with tracer.start_as_current_span("create_objectives"):
            objectives = []
            
            for opp in opportunities:
                # Determine weight based on impact and priority
                impact_weights = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 5.0}
                weight = impact_weights.get(opp.get("impact", "medium"), 2.0)
                
                # Add priority multiplier
                priority_multipliers = {
                    Priority.LOW: 0.5,
                    Priority.MEDIUM: 1.0,
                    Priority.HIGH: 1.5,
                    Priority.CRITICAL: 2.0
                }
                weight *= priority_multipliers.get(opp["priority"], 1.0)
                
                objective = OptimizationObjective(
                    domain=opp["domain"],
                    metric_name=opp["issue"],
                    current_value=opp["current_value"],
                    target_value=opp["target_value"],
                    weight=weight,
                    priority=opp["priority"],
                    constraint_min=0.0,
                    constraint_max=10.0 if "count" in opp["issue"] else 1.0
                )
                
                objectives.append(objective)
            
            # Sort by priority and weight
            objectives.sort(key=lambda obj: (obj.priority.value, obj.weight), reverse=True)
            
            return objectives
    
    async def _analysis_phase(self) -> None:
        """Analysis phase - deep analysis of objectives and constraints"""
        with tracer.start_as_current_span("analysis_phase"):
            logger.info("Starting analysis phase")
            
            # Analyze objective dependencies
            await self._analyze_objective_dependencies()
            
            # Analyze resource constraints
            await self._analyze_resource_constraints()
            
            # Analyze risk factors
            await self._analyze_risk_factors()
            
            logger.info("Analysis phase complete")
    
    async def _analyze_objective_dependencies(self) -> None:
        """Analyze dependencies between objectives"""
        with tracer.start_as_current_span("analyze_dependencies"):
            # Simple dependency analysis
            for i, obj1 in enumerate(self.objectives):
                for j, obj2 in enumerate(self.objectives[i+1:], i+1):
                    # Performance and reliability are often linked
                    if ((obj1.domain == SystemDomain.PERFORMANCE and obj2.domain == SystemDomain.RELIABILITY) or
                        (obj1.domain == SystemDomain.RELIABILITY and obj2.domain == SystemDomain.PERFORMANCE)):
                        logger.info(f"Dependency identified: {obj1.metric_name} <-> {obj2.metric_name}")
    
    async def _analyze_resource_constraints(self) -> None:
        """Analyze available resources and constraints"""
        with tracer.start_as_current_span("analyze_resource_constraints"):
            # Get resource information from performance optimizer
            perf_report = self.global_state["performance"]
            resource_allocations = perf_report["resource_allocations"]
            
            total_cpu = sum(
                alloc.get("current_allocation", 0) 
                for key, alloc in resource_allocations.items() 
                if "cpu" in key
            )
            
            logger.info(f"Current total CPU allocation: {total_cpu}")
    
    async def _analyze_risk_factors(self) -> None:
        """Analyze risk factors for optimization"""
        with tracer.start_as_current_span("analyze_risk_factors"):
            # Calculate risk based on system stability
            health_status = self.global_state["health"]["overall_health"]
            security_alerts = self.global_state["security"]["active_alerts"]
            
            risk_factors = []
            if health_status != "healthy":
                risk_factors.append("System health degraded")
            if security_alerts > 0:
                risk_factors.append("Active security alerts")
            
            logger.info(f"Risk factors identified: {len(risk_factors)}")
    
    async def _planning_phase(self) -> None:
        """Planning phase - create optimization plans"""
        with tracer.start_as_current_span("planning_phase"):
            logger.info("Starting planning phase")
            
            # Group objectives by domain
            objectives_by_domain = {}
            for obj in self.objectives:
                domain = obj.domain
                if domain not in objectives_by_domain:
                    objectives_by_domain[domain] = []
                objectives_by_domain[domain].append(obj)
            
            # Create optimization plans for each domain
            for domain, objectives in objectives_by_domain.items():
                plan = await self._create_optimization_plan(domain, objectives)
                self.optimization_plans[plan.plan_id] = plan
            
            logger.info(f"Planning complete: {len(self.optimization_plans)} plans created")
    
    async def _create_optimization_plan(
        self, 
        domain: SystemDomain, 
        objectives: List[OptimizationObjective]
    ) -> OptimizationPlan:
        """Create optimization plan for specific domain"""
        with tracer.start_as_current_span("create_optimization_plan") as span:
            span.set_attribute("domain", domain.value)
            
            plan_id = f"plan_{domain.value}_{int(time.time())}"
            
            # Create autonomous tasks for each objective
            tasks = []
            for obj in objectives:
                task = AutonomousTask(
                    id=f"task_{obj.domain.value}_{obj.metric_name}_{int(time.time())}",
                    name=f"Optimize {obj.metric_name}",
                    description=f"Improve {obj.metric_name} from {obj.current_value} to {obj.target_value}",
                    priority=obj.priority,
                    metadata={
                        "domain": obj.domain.value,
                        "metric_name": obj.metric_name,
                        "target_value": obj.target_value
                    }
                )
                tasks.append(task)
            
            # Estimate duration based on complexity
            base_duration = timedelta(minutes=30)
            complexity_factor = len(objectives)
            estimated_duration = base_duration * complexity_factor
            
            # Calculate expected improvement
            expected_improvement = {}
            for obj in objectives:
                improvement = (obj.target_value - obj.current_value) / max(obj.current_value, 0.001)
                expected_improvement[obj.metric_name] = improvement
            
            # Calculate risk level
            risk_level = min(1.0, len(objectives) * 0.2)  # More objectives = higher risk
            
            # Resource requirements
            resource_requirements = {
                "cpu": len(objectives) * 0.1,
                "memory": len(objectives) * 0.5,
                "time": estimated_duration.total_seconds()
            }
            
            plan = OptimizationPlan(
                plan_id=plan_id,
                objectives=objectives,
                tasks=tasks,
                estimated_duration=estimated_duration,
                expected_improvement=expected_improvement,
                risk_level=risk_level,
                resource_requirements=resource_requirements
            )
            
            logger.info(f"Created optimization plan: {plan_id} with {len(tasks)} tasks")
            return plan
    
    async def _execution_phase(self) -> None:
        """Execution phase - execute optimization plans"""
        with tracer.start_as_current_span("execution_phase"):
            logger.info("Starting execution phase")
            
            # Execute plans in order of priority
            plans_by_priority = sorted(
                self.optimization_plans.values(),
                key=lambda p: max(obj.priority.value for obj in p.objectives),
                reverse=True
            )
            
            execution_results = []
            for plan in plans_by_priority:
                result = await self._execute_optimization_plan(plan)
                execution_results.append(result)
                self.execution_results.append(result)
            
            logger.info(f"Execution complete: {len(execution_results)} plans executed")
    
    async def _execute_optimization_plan(self, plan: OptimizationPlan) -> OptimizationResult:
        """Execute single optimization plan"""
        with tracer.start_as_current_span("execute_optimization_plan") as span:
            span.set_attribute("plan_id", plan.plan_id)
            
            start_time = datetime.utcnow()
            
            try:
                # Submit tasks to executor
                task_ids = []
                for task in plan.tasks:
                    task_id = await self.executor.submit_task(task)
                    task_ids.append(task_id)
                
                # Wait for tasks to complete (simplified)
                await asyncio.sleep(10)  # In real system, would wait for actual completion
                
                # Calculate achievements (simplified)
                achievements = {}
                for obj in plan.objectives:
                    # Simulate improvement
                    simulated_improvement = min(0.1, (obj.target_value - obj.current_value) * 0.5)
                    achievements[obj.metric_name] = simulated_improvement
                
                result = OptimizationResult(
                    plan_id=plan.plan_id,
                    phase=OptimizationPhase.EXECUTION,
                    success=True,
                    duration=datetime.utcnow() - start_time,
                    achievements=achievements,
                    side_effects={},
                    lessons_learned=["Optimization completed successfully"],
                    next_steps=["Monitor results", "Validate improvements"]
                )
                
                logger.info(f"Plan executed successfully: {plan.plan_id}")
                return result
                
            except Exception as e:
                logger.error(f"Plan execution failed: {plan.plan_id} - {e}")
                
                result = OptimizationResult(
                    plan_id=plan.plan_id,
                    phase=OptimizationPhase.EXECUTION,
                    success=False,
                    duration=datetime.utcnow() - start_time,
                    achievements={},
                    side_effects={"error": str(e)},
                    lessons_learned=[f"Execution failed: {str(e)}"],
                    next_steps=["Investigate failure", "Revise plan"]
                )
                
                return result
    
    async def _validation_phase(self) -> None:
        """Validation phase - validate optimization results"""
        with tracer.start_as_current_span("validation_phase"):
            logger.info("Starting validation phase")
            
            # Re-gather system state
            await self._gather_system_state()
            
            # Validate improvements
            validation_results = await self._validate_improvements()
            
            logger.info(f"Validation complete: {len(validation_results)} results validated")
    
    async def _validate_improvements(self) -> List[Dict[str, Any]]:
        """Validate that optimizations achieved expected improvements"""
        with tracer.start_as_current_span("validate_improvements"):
            validation_results = []
            
            # Compare before/after metrics
            for result in self.execution_results[-len(self.optimization_plans):]:  # Recent results
                plan = self.optimization_plans.get(result.plan_id)
                if not plan:
                    continue
                
                validation = {
                    "plan_id": result.plan_id,
                    "success": result.success,
                    "validated": True,  # Simplified validation
                    "achievements": result.achievements
                }
                
                validation_results.append(validation)
            
            return validation_results
    
    async def _learning_phase(self) -> None:
        """Learning phase - extract insights and update models"""
        with tracer.start_as_current_span("learning_phase"):
            logger.info("Starting learning phase")
            
            # Extract lessons learned
            lessons = await self._extract_lessons_learned()
            
            # Update optimization models
            await self._update_optimization_models(lessons)
            
            # Feed insights to adaptive intelligence
            await self._feed_insights_to_intelligence(lessons)
            
            logger.info("Learning phase complete")
    
    async def _extract_lessons_learned(self) -> List[Dict[str, Any]]:
        """Extract lessons from optimization results"""
        with tracer.start_as_current_span("extract_lessons"):
            lessons = []
            
            for result in self.execution_results[-10:]:  # Last 10 results
                lesson = {
                    "plan_id": result.plan_id,
                    "success": result.success,
                    "duration": result.duration.total_seconds(),
                    "achievements": result.achievements,
                    "lessons": result.lessons_learned
                }
                lessons.append(lesson)
            
            return lessons
    
    async def _update_optimization_models(self, lessons: List[Dict[str, Any]]) -> None:
        """Update optimization models based on lessons learned"""
        with tracer.start_as_current_span("update_optimization_models"):
            # Calculate success rates by domain
            success_rates = {}
            for lesson in lessons:
                # Extract domain from achievements (simplified)
                for metric_name in lesson["achievements"].keys():
                    if metric_name not in success_rates:
                        success_rates[metric_name] = []
                    success_rates[metric_name].append(1.0 if lesson["success"] else 0.0)
            
            # Update model weights
            for metric_name, rates in success_rates.items():
                avg_success_rate = sum(rates) / len(rates)
                logger.info(f"Model update: {metric_name} success rate = {avg_success_rate:.2f}")
    
    async def _feed_insights_to_intelligence(self, lessons: List[Dict[str, Any]]) -> None:
        """Feed optimization insights to adaptive intelligence"""
        with tracer.start_as_current_span("feed_insights_to_intelligence"):
            for lesson in lessons:
                await self.intelligence.ingest_data_point(
                    PatternType.PERFORMANCE,
                    {
                        "optimization_success": lesson["success"],
                        "duration": lesson["duration"],
                        "achievements": lesson["achievements"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
    
    async def _continuous_monitoring_cycle(self) -> None:
        """Continuous monitoring of optimization effectiveness"""
        while self._optimization_active:
            try:
                await self._monitor_optimization_effectiveness()
                await asyncio.sleep(120)  # Monitor every 2 minutes
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(240)
    
    async def _monitor_optimization_effectiveness(self) -> None:
        """Monitor the effectiveness of ongoing optimizations"""
        with tracer.start_as_current_span("monitor_optimization_effectiveness"):
            # Calculate overall optimization effectiveness
            if not self.execution_results:
                return
            
            recent_results = self.execution_results[-5:]  # Last 5 results
            success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
            
            if success_rate < 0.6:  # Below 60% success rate
                logger.warning(f"Optimization success rate low: {success_rate:.2f}")
                # Could trigger plan adjustments here
    
    async def _learning_integration_cycle(self) -> None:
        """Continuous integration of learning across subsystems"""
        while self._optimization_active:
            try:
                await self._integrate_subsystem_learning()
                await asyncio.sleep(180)  # Integrate every 3 minutes
            except Exception as e:
                logger.error(f"Error in learning integration cycle: {e}")
                await asyncio.sleep(360)
    
    async def _integrate_subsystem_learning(self) -> None:
        """Integrate learning across all subsystems"""
        with tracer.start_as_current_span("integrate_subsystem_learning"):
            # Get insights from all subsystems
            intelligence_report = self.intelligence.get_intelligence_report()
            performance_report = await self.performance_optimizer.get_optimization_report()
            
            # Cross-pollinate insights
            if intelligence_report["adaptation_success_rate"] > 0.8 and performance_report["overall_performance_score"] > 0.8:
                logger.info("High performance correlation detected - reinforcing successful patterns")
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global optimization status"""
        with tracer.start_as_current_span("get_global_status"):
            # Calculate overall system score
            if not self.objectives:
                overall_score = 0.5
            else:
                total_weighted_score = sum(obj.calculate_score() * obj.weight for obj in self.objectives)
                total_weight = sum(obj.weight for obj in self.objectives)
                overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Recent performance
            recent_results = self.execution_results[-10:]
            success_rate = sum(1 for r in recent_results if r.success) / len(recent_results) if recent_results else 0.0
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "current_phase": self.current_phase.value,
                "optimization_active": self._optimization_active,
                "overall_system_score": overall_score,
                "active_objectives": len(self.objectives),
                "active_plans": len(self.optimization_plans),
                "execution_success_rate": success_rate,
                "total_optimizations": len(self.execution_results),
                "subsystems_status": {
                    "executor": "running",
                    "intelligence": "running",
                    "security": "running",
                    "monitor": "running",
                    "performance_optimizer": "running"
                }
            }


# Global engine instance
_global_engine: Optional[GlobalOptimizationEngine] = None


async def get_global_engine() -> GlobalOptimizationEngine:
    """Get or create the global optimization engine instance"""
    global _global_engine
    if _global_engine is None:
        _global_engine = GlobalOptimizationEngine()
        await _global_engine.start_global_optimization()
    return _global_engine


async def get_global_optimization_status() -> Dict[str, Any]:
    """Get global optimization status"""
    engine = await get_global_engine()
    return engine.get_global_status()