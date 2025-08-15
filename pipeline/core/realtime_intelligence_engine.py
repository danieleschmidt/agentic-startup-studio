"""
Real-time Intelligence Engine - Generation 2 Enhancement
Advanced real-time processing and decision making system
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import heapq

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of real-time events"""
    USER_ACTION = "user_action"
    SYSTEM_METRIC = "system_metric"
    PERFORMANCE_ALERT = "performance_alert"
    BUSINESS_EVENT = "business_event"
    SECURITY_EVENT = "security_event"
    PREDICTION_TRIGGER = "prediction_trigger"
    OPTIMIZATION_SIGNAL = "optimization_signal"


class Priority(str, Enum):
    """Event priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProcessingMode(str, Enum):
    """Event processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    ADAPTIVE = "adaptive"


@dataclass
class RealTimeEvent:
    """Real-time event with metadata"""
    event_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: Priority = Priority.MEDIUM
    source: str = "unknown"
    correlation_id: Optional[str] = None
    processing_deadline: Optional[datetime] = None
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set processing deadline based on priority"""
        if self.processing_deadline is None:
            if self.priority == Priority.CRITICAL:
                self.processing_deadline = self.timestamp + timedelta(milliseconds=100)
            elif self.priority == Priority.HIGH:
                self.processing_deadline = self.timestamp + timedelta(milliseconds=500)
            elif self.priority == Priority.MEDIUM:
                self.processing_deadline = self.timestamp + timedelta(seconds=2)
            else:  # LOW
                self.processing_deadline = self.timestamp + timedelta(seconds=10)
    
    def is_expired(self) -> bool:
        """Check if event has expired based on deadline"""
        return datetime.utcnow() > self.processing_deadline
    
    def time_to_deadline_ms(self) -> float:
        """Get milliseconds until processing deadline"""
        delta = self.processing_deadline - datetime.utcnow()
        return max(0, delta.total_seconds() * 1000)


@dataclass
class ProcessingResult:
    """Result of event processing"""
    event_id: str
    success: bool
    processing_time_ms: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    actions_triggered: List[str] = field(default_factory=list)
    next_events: List[RealTimeEvent] = field(default_factory=list)


class EventProcessor:
    """Base class for event processors"""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
    async def process(self, event: RealTimeEvent) -> ProcessingResult:
        """Process a single event"""
        start_time = time.time()
        
        try:
            result_data = await self._process_event(event)
            processing_time = (time.time() - start_time) * 1000
            
            self.processed_count += 1
            self.total_processing_time += processing_time
            
            return ProcessingResult(
                event_id=event.event_id,
                success=True,
                processing_time_ms=processing_time,
                result_data=result_data,
                actions_triggered=self._get_triggered_actions(event, result_data)
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.error_count += 1
            
            return ProcessingResult(
                event_id=event.event_id,
                success=False,
                processing_time_ms=processing_time,
                error_message=str(e)
            )
    
    async def _process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Override this method to implement specific processing logic"""
        raise NotImplementedError
    
    def _get_triggered_actions(self, event: RealTimeEvent, result_data: Dict[str, Any]) -> List[str]:
        """Get list of actions triggered by this processing"""
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        avg_processing_time = (
            self.total_processing_time / self.processed_count 
            if self.processed_count > 0 else 0
        )
        
        return {
            "processor_id": self.processor_id,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.processed_count - self.error_count) / self.processed_count
                if self.processed_count > 0 else 0
            ),
            "avg_processing_time_ms": avg_processing_time
        }


class UserActionProcessor(EventProcessor):
    """Processor for user action events"""
    
    async def _process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Process user action event"""
        action = event.payload.get("action", "unknown")
        user_id = event.payload.get("user_id", "anonymous")
        
        # Analyze user behavior patterns
        behavior_analysis = await self._analyze_user_behavior(user_id, action)
        
        # Generate personalization insights
        personalization = await self._generate_personalization(user_id, behavior_analysis)
        
        return {
            "action_processed": action,
            "user_id": user_id,
            "behavior_analysis": behavior_analysis,
            "personalization": personalization,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_user_behavior(self, user_id: str, action: str) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        # Simulate behavior analysis
        return {
            "action_frequency": np.random.uniform(1, 10),
            "user_segment": np.random.choice(["new", "regular", "power_user"]),
            "engagement_score": np.random.uniform(0.3, 1.0),
            "predicted_next_action": np.random.choice(["view", "click", "purchase", "exit"])
        }
    
    async def _generate_personalization(self, user_id: str, behavior: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalization recommendations"""
        return {
            "recommended_content": f"content_for_{behavior.get('user_segment', 'regular')}",
            "optimal_timing": np.random.choice(["immediate", "5min", "1hour", "tomorrow"]),
            "confidence": np.random.uniform(0.6, 0.95)
        }
    
    def _get_triggered_actions(self, event: RealTimeEvent, result_data: Dict[str, Any]) -> List[str]:
        """Get triggered actions for user events"""
        actions = ["update_user_profile"]
        
        # Trigger personalization update if high engagement
        if result_data.get("behavior_analysis", {}).get("engagement_score", 0) > 0.8:
            actions.append("trigger_personalization_update")
        
        # Trigger retention campaign for new users
        if result_data.get("behavior_analysis", {}).get("user_segment") == "new":
            actions.append("trigger_onboarding_sequence")
        
        return actions


class PerformanceMetricProcessor(EventProcessor):
    """Processor for performance metric events"""
    
    def __init__(self, processor_id: str):
        super().__init__(processor_id)
        self.metric_thresholds = {
            "response_time": 1000,  # ms
            "cpu_usage": 80,  # percent
            "memory_usage": 85,  # percent
            "error_rate": 5  # percent
        }
    
    async def _process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Process performance metric event"""
        metric_name = event.payload.get("metric_name", "unknown")
        metric_value = event.payload.get("value", 0)
        
        # Check thresholds
        threshold_analysis = await self._analyze_thresholds(metric_name, metric_value)
        
        # Predict trend
        trend_analysis = await self._analyze_trend(metric_name, metric_value)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            metric_name, metric_value, threshold_analysis, trend_analysis
        )
        
        return {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold_analysis": threshold_analysis,
            "trend_analysis": trend_analysis,
            "recommendations": recommendations,
            "severity": threshold_analysis.get("severity", "normal")
        }
    
    async def _analyze_thresholds(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Analyze metric against thresholds"""
        threshold = self.metric_thresholds.get(metric_name)
        if threshold is None:
            return {"status": "no_threshold", "severity": "normal"}
        
        percentage = (value / threshold) * 100
        
        if percentage >= 95:
            severity = "critical"
        elif percentage >= 85:
            severity = "high"
        elif percentage >= 70:
            severity = "medium"
        else:
            severity = "normal"
        
        return {
            "status": "above_threshold" if percentage >= 100 else "within_threshold",
            "percentage": percentage,
            "severity": severity,
            "threshold": threshold
        }
    
    async def _analyze_trend(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Analyze metric trend"""
        # Simulate trend analysis (in real system would use historical data)
        trend_direction = np.random.choice(["increasing", "decreasing", "stable"])
        trend_strength = np.random.uniform(0.1, 0.9)
        
        return {
            "direction": trend_direction,
            "strength": trend_strength,
            "forecast_5min": value * (1 + np.random.uniform(-0.1, 0.1)),
            "forecast_15min": value * (1 + np.random.uniform(-0.2, 0.2))
        }
    
    async def _generate_recommendations(
        self, 
        metric_name: str, 
        value: float, 
        threshold_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if threshold_analysis.get("severity") == "critical":
            if metric_name == "response_time":
                recommendations.extend([
                    "scale_up_instances",
                    "enable_caching",
                    "optimize_database_queries"
                ])
            elif metric_name == "memory_usage":
                recommendations.extend([
                    "increase_memory_allocation",
                    "trigger_garbage_collection",
                    "identify_memory_leaks"
                ])
        
        if trend_analysis.get("direction") == "increasing" and trend_analysis.get("strength") > 0.7:
            recommendations.append("schedule_preventive_maintenance")
        
        return recommendations
    
    def _get_triggered_actions(self, event: RealTimeEvent, result_data: Dict[str, Any]) -> List[str]:
        """Get triggered actions for performance events"""
        actions = []
        severity = result_data.get("severity", "normal")
        
        if severity == "critical":
            actions.extend([
                "send_alert",
                "trigger_auto_scaling",
                "log_incident"
            ])
        elif severity == "high":
            actions.extend([
                "send_warning",
                "schedule_investigation"
            ])
        
        return actions


class SecurityEventProcessor(EventProcessor):
    """Processor for security events"""
    
    async def _process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Process security event"""
        threat_type = event.payload.get("threat_type", "unknown")
        source_ip = event.payload.get("source_ip", "unknown")
        
        # Threat analysis
        threat_analysis = await self._analyze_threat(threat_type, source_ip)
        
        # Risk assessment
        risk_assessment = await self._assess_risk(threat_analysis)
        
        # Response plan
        response_plan = await self._generate_response_plan(threat_analysis, risk_assessment)
        
        return {
            "threat_type": threat_type,
            "source_ip": source_ip,
            "threat_analysis": threat_analysis,
            "risk_assessment": risk_assessment,
            "response_plan": response_plan
        }
    
    async def _analyze_threat(self, threat_type: str, source_ip: str) -> Dict[str, Any]:
        """Analyze security threat"""
        # Simulate threat intelligence lookup
        threat_score = np.random.uniform(0.1, 1.0)
        is_known_bad = np.random.choice([True, False], p=[0.2, 0.8])
        
        return {
            "threat_score": threat_score,
            "is_known_bad_ip": is_known_bad,
            "geolocation": np.random.choice(["US", "CN", "RU", "Unknown"]),
            "attack_pattern": np.random.choice(["brute_force", "sql_injection", "xss", "ddos"])
        }
    
    async def _assess_risk(self, threat_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level"""
        threat_score = threat_analysis.get("threat_score", 0)
        is_known_bad = threat_analysis.get("is_known_bad_ip", False)
        
        if is_known_bad or threat_score > 0.8:
            risk_level = "high"
        elif threat_score > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "confidence": np.random.uniform(0.7, 0.95),
            "impact_assessment": np.random.choice(["low", "medium", "high"]),
            "urgency": np.random.choice(["low", "medium", "high", "critical"])
        }
    
    async def _generate_response_plan(
        self, 
        threat_analysis: Dict[str, Any], 
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate security response plan"""
        risk_level = risk_assessment.get("risk_level", "low")
        
        if risk_level == "high":
            actions = ["block_ip", "alert_security_team", "initiate_incident_response"]
        elif risk_level == "medium":
            actions = ["rate_limit_ip", "increase_monitoring", "alert_administrators"]
        else:
            actions = ["log_event", "continue_monitoring"]
        
        return {
            "immediate_actions": actions,
            "escalation_path": ["l1_security", "l2_security", "security_manager"],
            "estimated_resolution_time": np.random.randint(5, 120)  # minutes
        }
    
    def _get_triggered_actions(self, event: RealTimeEvent, result_data: Dict[str, Any]) -> List[str]:
        """Get triggered actions for security events"""
        risk_level = result_data.get("risk_assessment", {}).get("risk_level", "low")
        actions = ["update_security_log"]
        
        if risk_level == "high":
            actions.extend([
                "trigger_security_alert",
                "block_source_ip",
                "notify_security_team"
            ])
        
        return actions


class EventStream:
    """High-performance event stream with priority queue"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.events: List[Tuple[float, RealTimeEvent]] = []  # Priority queue
        self.event_count = 0
        self.dropped_count = 0
        self.lock = threading.Lock()
        
    def add_event(self, event: RealTimeEvent) -> bool:
        """Add event to stream with priority ordering"""
        with self.lock:
            if len(self.events) >= self.max_size:
                # Drop lowest priority event if at capacity
                if self.events and self._get_priority_score(event) > self.events[0][0]:
                    heapq.heappop(self.events)
                    self.dropped_count += 1
                else:
                    self.dropped_count += 1
                    return False
            
            priority_score = self._get_priority_score(event)
            heapq.heappush(self.events, (priority_score, event))
            self.event_count += 1
            return True
    
    def get_next_event(self) -> Optional[RealTimeEvent]:
        """Get highest priority event from stream"""
        with self.lock:
            if self.events:
                _, event = heapq.heappop(self.events)
                return event
            return None
    
    def peek_next_event(self) -> Optional[RealTimeEvent]:
        """Peek at highest priority event without removing"""
        with self.lock:
            if self.events:
                return self.events[0][1]
            return None
    
    def get_events_by_deadline(self, max_events: int = 100) -> List[RealTimeEvent]:
        """Get events sorted by processing deadline"""
        with self.lock:
            # Extract events and sort by deadline
            events_with_deadline = []
            remaining_events = []
            
            while self.events and len(events_with_deadline) < max_events:
                priority_score, event = heapq.heappop(self.events)
                if not event.is_expired():
                    events_with_deadline.append(event)
                # Expired events are dropped
            
            # Put remaining events back
            while self.events:
                remaining_events.append(heapq.heappop(self.events))
            
            for priority_score, event in remaining_events:
                if not event.is_expired():
                    heapq.heappush(self.events, (priority_score, event))
            
            # Sort by deadline
            events_with_deadline.sort(key=lambda e: e.processing_deadline)
            return events_with_deadline
    
    def _get_priority_score(self, event: RealTimeEvent) -> float:
        """Calculate priority score for event (lower = higher priority)"""
        base_scores = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 2.0,
            Priority.MEDIUM: 3.0,
            Priority.LOW: 4.0
        }
        
        priority_score = base_scores.get(event.priority, 3.0)
        
        # Adjust based on time to deadline
        time_to_deadline = event.time_to_deadline_ms()
        urgency_factor = max(0.1, time_to_deadline / 10000)  # Normalize to 0.1-1.0
        
        return priority_score * urgency_factor
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        with self.lock:
            return {
                "current_size": len(self.events),
                "max_size": self.max_size,
                "total_events": self.event_count,
                "dropped_events": self.dropped_count,
                "utilization": len(self.events) / self.max_size
            }


class RealTimeIntelligenceEngine:
    """
    Main real-time intelligence processing engine
    """
    
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.event_stream = EventStream()
        self.processors: Dict[EventType, EventProcessor] = {}
        self.processing_stats: Dict[str, Any] = defaultdict(int)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queues: Dict[Priority, asyncio.Queue] = {}
        self.result_handlers: List[Callable[[ProcessingResult], None]] = []
        self.active_processors: Set[str] = set()
        self._engine_active = True
        
        # Initialize processing queues
        for priority in Priority:
            self.processing_queues[priority] = asyncio.Queue(maxsize=1000)
    
    async def initialize(self) -> None:
        """Initialize the real-time intelligence engine"""
        with tracer.start_as_current_span("initialize_realtime_engine"):
            logger.info("Initializing Real-time Intelligence Engine")
            
            # Initialize event processors
            await self._initialize_processors()
            
            # Start processing loops
            asyncio.create_task(self._event_ingestion_loop())
            asyncio.create_task(self._event_distribution_loop())
            
            # Start priority-based processing loops
            for priority in Priority:
                asyncio.create_task(self._priority_processing_loop(priority))
            
            # Start monitoring and optimization loops
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._adaptive_optimization_loop())
            
            logger.info(f"Real-time Intelligence Engine initialized with {len(self.processors)} processors")
    
    async def _initialize_processors(self) -> None:
        """Initialize event processors for different event types"""
        self.processors[EventType.USER_ACTION] = UserActionProcessor("user_action_processor")
        self.processors[EventType.SYSTEM_METRIC] = PerformanceMetricProcessor("performance_processor")
        self.processors[EventType.SECURITY_EVENT] = SecurityEventProcessor("security_processor")
        
        # Generic processor for other event types
        class GenericProcessor(EventProcessor):
            async def _process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
                return {"processed": True, "event_type": event.event_type.value}
        
        for event_type in EventType:
            if event_type not in self.processors:
                self.processors[event_type] = GenericProcessor(f"{event_type.value}_processor")
    
    async def process_event(self, event: RealTimeEvent) -> bool:
        """Add event to processing stream"""
        with tracer.start_as_current_span("process_event") as span:
            span.set_attributes({
                "event_type": event.event_type.value,
                "priority": event.priority.value,
                "event_id": event.event_id
            })
            
            # Add to event stream
            success = self.event_stream.add_event(event)
            
            if success:
                self.processing_stats["events_received"] += 1
                logger.debug(f"Event {event.event_id} added to stream")
            else:
                self.processing_stats["events_dropped"] += 1
                logger.warning(f"Event {event.event_id} dropped - stream full")
            
            return success
    
    async def _event_ingestion_loop(self) -> None:
        """Main event ingestion loop"""
        while self._engine_active:
            try:
                # Get events from stream and distribute to priority queues
                events = self.event_stream.get_events_by_deadline(100)
                
                for event in events:
                    if not event.is_expired():
                        try:
                            await self.processing_queues[event.priority].put(event)
                        except asyncio.QueueFull:
                            self.processing_stats["queue_full_drops"] += 1
                            logger.warning(f"Priority queue {event.priority.value} full, dropping event")
                    else:
                        self.processing_stats["expired_events"] += 1
                
                await asyncio.sleep(0.001)  # 1ms loop
                
            except Exception as e:
                logger.error(f"Error in event ingestion loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _event_distribution_loop(self) -> None:
        """Distribute events from stream to priority queues"""
        while self._engine_active:
            try:
                event = self.event_stream.get_next_event()
                if event:
                    if not event.is_expired():
                        await self.processing_queues[event.priority].put(event)
                    else:
                        self.processing_stats["expired_events"] += 1
                else:
                    await asyncio.sleep(0.001)  # No events, short sleep
                    
            except Exception as e:
                logger.error(f"Error in event distribution loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _priority_processing_loop(self, priority: Priority) -> None:
        """Processing loop for specific priority level"""
        queue = self.processing_queues[priority]
        
        # Different batch sizes based on priority
        batch_sizes = {
            Priority.CRITICAL: 1,   # Process immediately
            Priority.HIGH: 5,       # Small batches
            Priority.MEDIUM: 10,    # Medium batches
            Priority.LOW: 20        # Larger batches
        }
        batch_size = batch_sizes.get(priority, 10)
        
        while self._engine_active:
            try:
                # Collect batch of events
                events = []
                timeout = 0.001 if priority == Priority.CRITICAL else 0.01
                
                for _ in range(batch_size):
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=timeout)
                        events.append(event)
                    except asyncio.TimeoutError:
                        break
                
                if events:
                    await self._process_event_batch(events)
                else:
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error in {priority.value} processing loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _process_event_batch(self, events: List[RealTimeEvent]) -> None:
        """Process a batch of events"""
        with tracer.start_as_current_span("process_event_batch") as span:
            span.set_attribute("batch_size", len(events))
            
            # Process events concurrently
            tasks = []
            for event in events:
                processor = self.processors.get(event.event_type)
                if processor:
                    task = asyncio.create_task(self._process_single_event(event, processor))
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing event {events[i].event_id}: {result}")
                        self.processing_stats["processing_errors"] += 1
                    elif isinstance(result, ProcessingResult):
                        await self._handle_processing_result(result)
    
    async def _process_single_event(self, event: RealTimeEvent, processor: EventProcessor) -> ProcessingResult:
        """Process a single event with the appropriate processor"""
        processor_id = f"{processor.processor_id}_{event.event_id}"
        self.active_processors.add(processor_id)
        
        try:
            result = await processor.process(event)
            self.processing_stats["events_processed"] += 1
            
            if result.success:
                self.processing_stats["successful_processing"] += 1
            else:
                self.processing_stats["failed_processing"] += 1
            
            return result
            
        finally:
            self.active_processors.discard(processor_id)
    
    async def _handle_processing_result(self, result: ProcessingResult) -> None:
        """Handle the result of event processing"""
        # Execute triggered actions
        for action in result.actions_triggered:
            await self._execute_action(action, result)
        
        # Process any generated follow-up events
        for next_event in result.next_events:
            await self.process_event(next_event)
        
        # Notify result handlers
        for handler in self.result_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Error in result handler: {e}")
    
    async def _execute_action(self, action: str, result: ProcessingResult) -> None:
        """Execute an action triggered by event processing"""
        with tracer.start_as_current_span("execute_action") as span:
            span.set_attributes({
                "action": action,
                "event_id": result.event_id
            })
            
            # Simulate action execution
            if action == "send_alert":
                await self._send_alert(result)
            elif action == "trigger_auto_scaling":
                await self._trigger_auto_scaling(result)
            elif action == "update_user_profile":
                await self._update_user_profile(result)
            elif action == "block_source_ip":
                await self._block_ip(result)
            else:
                logger.debug(f"Executed action: {action}")
    
    async def _send_alert(self, result: ProcessingResult) -> None:
        """Send alert based on processing result"""
        logger.info(f"ALERT: Processing result for event {result.event_id}")
        # In real system, would integrate with alerting service
    
    async def _trigger_auto_scaling(self, result: ProcessingResult) -> None:
        """Trigger auto-scaling based on metrics"""
        logger.info(f"AUTO-SCALING: Triggered for event {result.event_id}")
        # In real system, would integrate with container orchestration
    
    async def _update_user_profile(self, result: ProcessingResult) -> None:
        """Update user profile based on behavior"""
        logger.debug(f"USER_PROFILE: Updated for event {result.event_id}")
        # In real system, would update user database
    
    async def _block_ip(self, result: ProcessingResult) -> None:
        """Block IP address for security"""
        logger.warning(f"SECURITY: IP blocked for event {result.event_id}")
        # In real system, would update firewall rules
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor engine performance and adjust accordingly"""
        while self._engine_active:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics"""
        # Queue utilization
        queue_stats = {}
        for priority, queue in self.processing_queues.items():
            queue_stats[priority.value] = {
                "size": queue.qsize(),
                "max_size": queue.maxsize,
                "utilization": queue.qsize() / queue.maxsize
            }
        
        # Stream stats
        stream_stats = self.event_stream.get_stats()
        
        # Processor stats
        processor_stats = {}
        for event_type, processor in self.processors.items():
            processor_stats[event_type.value] = processor.get_stats()
        
        # Log performance summary
        if any(stats["utilization"] > 0.8 for stats in queue_stats.values()):
            logger.warning("High queue utilization detected")
        
        if stream_stats["utilization"] > 0.9:
            logger.warning("Event stream near capacity")
    
    async def _adaptive_optimization_loop(self) -> None:
        """Adaptively optimize processing based on load patterns"""
        while self._engine_active:
            try:
                await self._optimize_processing()
                await asyncio.sleep(60)  # Optimize every minute
            except Exception as e:
                logger.error(f"Error in adaptive optimization: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_processing(self) -> None:
        """Optimize processing parameters based on current load"""
        # Analyze queue utilization patterns
        high_utilization_queues = []
        for priority, queue in self.processing_queues.items():
            utilization = queue.qsize() / queue.maxsize
            if utilization > 0.7:
                high_utilization_queues.append(priority)
        
        # Adaptive queue size adjustment
        if len(high_utilization_queues) > 1:
            logger.info("High load detected, optimizing queue processing")
            # In real system, would adjust worker allocation or batch sizes
    
    def add_result_handler(self, handler: Callable[[ProcessingResult], None]) -> None:
        """Add a result handler for processing results"""
        self.result_handlers.append(handler)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        with tracer.start_as_current_span("engine_status"):
            # Processing statistics
            total_events = self.processing_stats.get("events_received", 0)
            processed_events = self.processing_stats.get("events_processed", 0)
            
            processing_rate = (
                processed_events / max(1, total_events) * 100
                if total_events > 0 else 0
            )
            
            # Queue status
            queue_status = {}
            for priority, queue in self.processing_queues.items():
                queue_status[priority.value] = {
                    "current_size": queue.qsize(),
                    "max_size": queue.maxsize,
                    "utilization_percent": (queue.qsize() / queue.maxsize) * 100
                }
            
            # Processor status
            processor_status = {}
            for event_type, processor in self.processors.items():
                processor_status[event_type.value] = processor.get_stats()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "engine_active": self._engine_active,
                "active_processors": len(self.active_processors),
                "max_workers": self.max_workers,
                "processing_statistics": dict(self.processing_stats),
                "processing_rate_percent": processing_rate,
                "event_stream": self.event_stream.get_stats(),
                "queue_status": queue_status,
                "processor_status": processor_status,
                "result_handlers": len(self.result_handlers)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the real-time intelligence engine"""
        logger.info("Shutting down Real-time Intelligence Engine")
        self._engine_active = False
        
        # Wait for active processors to complete
        max_wait_time = 30  # seconds
        wait_start = time.time()
        
        while self.active_processors and (time.time() - wait_start) < max_wait_time:
            await asyncio.sleep(0.1)
        
        if self.active_processors:
            logger.warning(f"Shutdown with {len(self.active_processors)} processors still active")
        
        self.executor.shutdown(wait=True)


# Global engine instance
_realtime_engine: Optional[RealTimeIntelligenceEngine] = None


async def get_realtime_engine() -> RealTimeIntelligenceEngine:
    """Get or create the global real-time intelligence engine"""
    global _realtime_engine
    if _realtime_engine is None:
        _realtime_engine = RealTimeIntelligenceEngine()
        await _realtime_engine.initialize()
    return _realtime_engine


async def process_realtime_event(
    event_type: EventType,
    payload: Dict[str, Any],
    priority: Priority = Priority.MEDIUM,
    source: str = "system"
) -> bool:
    """Convenience function to process a real-time event"""
    engine = await get_realtime_engine()
    
    event = RealTimeEvent(
        event_id=f"{event_type.value}_{int(time.time() * 1000)}",
        event_type=event_type,
        payload=payload,
        priority=priority,
        source=source
    )
    
    return await engine.process_event(event)


async def process_user_action(action: str, user_id: str, context: Dict[str, Any] = None) -> bool:
    """Convenience function to process user action"""
    payload = {
        "action": action,
        "user_id": user_id,
        "context": context or {}
    }
    
    return await process_realtime_event(
        EventType.USER_ACTION,
        payload,
        Priority.MEDIUM,
        "user_interface"
    )


async def process_performance_metric(metric_name: str, value: float, context: Dict[str, Any] = None) -> bool:
    """Convenience function to process performance metric"""
    payload = {
        "metric_name": metric_name,
        "value": value,
        "context": context or {}
    }
    
    # Determine priority based on metric value
    priority = Priority.MEDIUM
    if metric_name in ["response_time", "error_rate"] and value > 1000:
        priority = Priority.HIGH
    elif metric_name in ["cpu_usage", "memory_usage"] and value > 90:
        priority = Priority.CRITICAL
    
    return await process_realtime_event(
        EventType.SYSTEM_METRIC,
        payload,
        priority,
        "monitoring_system"
    )