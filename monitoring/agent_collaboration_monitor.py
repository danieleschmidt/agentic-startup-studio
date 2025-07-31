"""
Advanced Agent Collaboration Monitoring and Performance Benchmarking
Provides comprehensive monitoring of multi-agent workflows and collaboration patterns
"""

import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import uuid
import statistics

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    ERROR = "error"
    TERMINATED = "terminated"


class InteractionType(Enum):
    """Types of agent interactions"""
    MESSAGE_PASSING = "message_passing"
    SHARED_MEMORY = "shared_memory"
    WORKFLOW_HANDOFF = "workflow_handoff"
    COLLABORATION = "collaboration"
    CONFLICT_RESOLUTION = "conflict_resolution"
    RESOURCE_SHARING = "resource_sharing"


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    agent_id: str
    agent_type: str
    timestamp: datetime
    
    # Performance metrics
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    
    # AI-specific metrics
    tokens_processed: int
    api_calls_made: int
    cost_incurred_usd: float
    quality_score: float
    
    # Collaboration metrics
    messages_sent: int
    messages_received: int
    collaborations_initiated: int
    collaborations_completed: int
    
    # Business metrics
    tasks_completed: int
    business_value_generated: float
    user_satisfaction_score: float


@dataclass
class AgentInteraction:
    """Agent-to-agent interaction record"""
    interaction_id: str
    source_agent: str
    target_agent: str
    interaction_type: InteractionType
    timestamp: datetime
    duration_ms: float
    success: bool
    payload_size_kb: float
    metadata: Dict[str, Any]


@dataclass
class WorkflowMetrics:
    """Multi-agent workflow performance metrics"""
    workflow_id: str
    workflow_type: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Participating agents
    agents: List[str]
    
    # Performance metrics
    total_duration_ms: float
    stages_completed: int
    stages_failed: int
    success_rate: float
    
    # Collaboration metrics
    total_interactions: int
    parallel_execution_time_ms: float
    sequential_execution_time_ms: float
    collaboration_efficiency: float  # parallel / (parallel + sequential)
    
    # Resource utilization
    peak_memory_usage_mb: float
    total_cpu_time_ms: float
    total_api_calls: int
    total_cost_usd: float
    
    # Quality metrics
    output_quality_score: float
    error_rate: float
    retry_count: int


class AgentCollaborationMonitor:
    """Advanced monitoring system for agent collaboration and performance"""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        metrics_retention_hours: int = 24,
        enable_prometheus: bool = True,
        prometheus_port: int = 8090
    ):
        self.monitoring_interval = monitoring_interval
        self.metrics_retention_hours = metrics_retention_hours
        self.enable_prometheus = enable_prometheus
        
        # Data storage
        self.agent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.interactions: deque = deque(maxlen=50000)
        self.workflows: Dict[str, WorkflowMetrics] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.collaboration_graph = nx.MultiDiGraph()
        
        # Performance tracking
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Prometheus metrics
        if enable_prometheus:
            self._setup_prometheus_metrics()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Collaboration analysis
        self.collaboration_patterns: Dict[str, Dict[str, Any]] = {}
        self.bottleneck_detection: Dict[str, List[str]] = defaultdict(list)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for agent monitoring"""
        
        # Agent performance metrics
        self.agent_processing_time = Histogram(
            'agent_processing_time_seconds',
            'Agent processing time in seconds',
            ['agent_id', 'agent_type', 'task_type']
        )
        
        self.agent_memory_usage = Gauge(
            'agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_id', 'agent_type']
        )
        
        self.agent_cpu_usage = Gauge(
            'agent_cpu_usage_percent',
            'Agent CPU usage percentage',
            ['agent_id', 'agent_type']
        )
        
        self.agent_success_rate = Gauge(
            'agent_success_rate',
            'Agent task success rate',
            ['agent_id', 'agent_type']
        )
        
        self.agent_error_count = Counter(
            'agent_errors_total',
            'Total agent errors',
            ['agent_id', 'agent_type', 'error_type']
        )
        
        # AI-specific metrics
        self.agent_tokens_processed = Counter(
            'agent_tokens_processed_total',
            'Total tokens processed by agent',
            ['agent_id', 'agent_type', 'model']
        )
        
        self.agent_api_calls = Counter(
            'agent_api_calls_total',
            'Total API calls made by agent',
            ['agent_id', 'agent_type', 'api_provider']
        )
        
        self.agent_cost = Counter(
            'agent_cost_usd_total',
            'Total cost incurred by agent in USD',
            ['agent_id', 'agent_type']
        )
        
        # Collaboration metrics
        self.agent_interactions = Counter(
            'agent_interactions_total',
            'Total agent interactions',
            ['source_agent', 'target_agent', 'interaction_type']
        )
        
        self.collaboration_efficiency = Gauge(
            'workflow_collaboration_efficiency',
            'Workflow collaboration efficiency ratio',
            ['workflow_type']
        )
        
        # Workflow metrics
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration',
            ['workflow_type', 'status']
        )
        
        self.workflow_agent_count = Histogram(
            'workflow_agent_count',
            'Number of agents in workflow',
            ['workflow_type']
        )
        
        # Business metrics
        self.business_value_generated = Counter(
            'business_value_generated_total',
            'Total business value generated',
            ['agent_type', 'value_type']
        )
        
        # Start Prometheus metrics server
        try:
            start_http_server(8090)
            print(f"Prometheus metrics server started on port 8090")
        except Exception as e:
            print(f"Warning: Could not start Prometheus server: {e}")
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("Agent collaboration monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        print("Agent collaboration monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update collaboration graph
                self._update_collaboration_graph()
                
                # Detect bottlenecks
                self._detect_bottlenecks()
                
                # Analyze collaboration patterns
                self._analyze_collaboration_patterns()
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self._update_prometheus_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def record_agent_metrics(self, metrics: AgentMetrics):
        """Record agent performance metrics"""
        
        # Store metrics
        self.agent_metrics[metrics.agent_id].append(metrics)
        
        # Update agent state based on metrics
        if metrics.error_count > 0:
            self.agent_states[metrics.agent_id] = AgentState.ERROR
        elif metrics.processing_time_ms > 0:
            self.agent_states[metrics.agent_id] = AgentState.ACTIVE
        else:
            self.agent_states[metrics.agent_id] = AgentState.IDLE
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.agent_processing_time.labels(
                agent_id=metrics.agent_id,
                agent_type=metrics.agent_type,
                task_type="general"
            ).observe(metrics.processing_time_ms / 1000)
            
            self.agent_memory_usage.labels(
                agent_id=metrics.agent_id,
                agent_type=metrics.agent_type
            ).set(metrics.memory_usage_mb * 1024 * 1024)
            
            self.agent_success_rate.labels(
                agent_id=metrics.agent_id,
                agent_type=metrics.agent_type
            ).set(metrics.success_rate)
    
    def record_agent_interaction(self, interaction: AgentInteraction):
        """Record agent-to-agent interaction"""
        
        # Store interaction
        self.interactions.append(interaction)
        
        # Update collaboration graph
        self.collaboration_graph.add_edge(
            interaction.source_agent,
            interaction.target_agent,
            interaction_type=interaction.interaction_type.value,
            timestamp=interaction.timestamp,
            duration=interaction.duration_ms,
            success=interaction.success
        )
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.agent_interactions.labels(
                source_agent=interaction.source_agent,
                target_agent=interaction.target_agent,
                interaction_type=interaction.interaction_type.value
            ).inc()
    
    def start_workflow_monitoring(
        self,
        workflow_id: str,
        workflow_type: str,
        participating_agents: List[str]
    ) -> str:
        """Start monitoring a multi-agent workflow"""
        
        workflow_metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=datetime.now(),
            end_time=None,
            agents=participating_agents,
            total_duration_ms=0,
            stages_completed=0,
            stages_failed=0,
            success_rate=0,
            total_interactions=0,
            parallel_execution_time_ms=0,
            sequential_execution_time_ms=0,
            collaboration_efficiency=0,
            peak_memory_usage_mb=0,
            total_cpu_time_ms=0,
            total_api_calls=0,
            total_cost_usd=0,
            output_quality_score=0,
            error_rate=0,
            retry_count=0
        )
        
        self.workflows[workflow_id] = workflow_metrics
        return workflow_id
    
    def complete_workflow_monitoring(
        self,
        workflow_id: str,
        success: bool,
        output_quality_score: float = 0.0,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """Complete workflow monitoring and calculate final metrics"""
        
        if workflow_id not in self.workflows:
            print(f"Warning: Workflow {workflow_id} not found in monitoring")
            return
        
        workflow = self.workflows[workflow_id]
        workflow.end_time = datetime.now()
        workflow.total_duration_ms = (workflow.end_time - workflow.start_time).total_seconds() * 1000
        workflow.output_quality_score = output_quality_score
        
        # Calculate collaboration metrics
        workflow_interactions = [
            i for i in self.interactions
            if i.timestamp >= workflow.start_time and
            (i.source_agent in workflow.agents or i.target_agent in workflow.agents)
        ]
        
        workflow.total_interactions = len(workflow_interactions)
        
        # Analyze parallel vs sequential execution
        self._analyze_workflow_execution_pattern(workflow_id)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            status = "success" if success else "failure"
            self.workflow_duration.labels(
                workflow_type=workflow.workflow_type,
                status=status
            ).observe(workflow.total_duration_ms / 1000)
            
            self.workflow_agent_count.labels(
                workflow_type=workflow.workflow_type
            ).observe(len(workflow.agents))
            
            self.collaboration_efficiency.labels(
                workflow_type=workflow.workflow_type
            ).set(workflow.collaboration_efficiency)
    
    def _analyze_workflow_execution_pattern(self, workflow_id: str):
        """Analyze workflow execution pattern to determine parallel vs sequential execution"""
        
        workflow = self.workflows[workflow_id]
        agent_timelines = defaultdict(list)
        
        # Build timeline for each agent
        for agent_id in workflow.agents:
            agent_metrics_list = list(self.agent_metrics.get(agent_id, []))
            workflow_metrics = [
                m for m in agent_metrics_list
                if workflow.start_time <= m.timestamp <= (workflow.end_time or datetime.now())
            ]
            
            for metrics in workflow_metrics:
                if metrics.processing_time_ms > 0:
                    agent_timelines[agent_id].append({
                        'start': metrics.timestamp,
                        'duration': metrics.processing_time_ms
                    })
        
        # Calculate parallel vs sequential time
        all_periods = []
        for agent_id, periods in agent_timelines.items():
            for period in periods:
                all_periods.append({
                    'start': period['start'].timestamp(),
                    'end': period['start'].timestamp() + (period['duration'] / 1000),
                    'agent': agent_id
                })
        
        # Sort by start time
        all_periods.sort(key=lambda x: x['start'])
        
        # Calculate overlapping periods (parallel execution)
        parallel_time = 0
        sequential_time = 0
        
        for i, period in enumerate(all_periods):
            # Check for overlaps with subsequent periods
            overlapping = False
            for j in range(i + 1, len(all_periods)):
                other_period = all_periods[j]
                if (other_period['start'] < period['end'] and 
                    other_period['agent'] != period['agent']):
                    overlapping = True
                    break
            
            duration = period['end'] - period['start']
            if overlapping:
                parallel_time += duration
            else:
                sequential_time += duration
        
        # Update workflow metrics
        workflow.parallel_execution_time_ms = parallel_time * 1000
        workflow.sequential_execution_time_ms = sequential_time * 1000
        
        total_execution_time = parallel_time + sequential_time
        if total_execution_time > 0:
            workflow.collaboration_efficiency = parallel_time / total_execution_time
        else:
            workflow.collaboration_efficiency = 0
    
    def _update_collaboration_graph(self):
        """Update the collaboration graph with recent interactions"""
        
        # Add nodes for all active agents
        for agent_id, state in self.agent_states.items():
            if agent_id not in self.collaboration_graph:
                self.collaboration_graph.add_node(agent_id, state=state.value)
            else:
                self.collaboration_graph.nodes[agent_id]['state'] = state.value
    
    def _detect_bottlenecks(self):
        """Detect collaboration bottlenecks and performance issues"""
        
        current_time = datetime.now()
        recent_window = current_time - timedelta(minutes=5)
        
        # Analyze recent interactions for bottlenecks
        recent_interactions = [
            i for i in self.interactions
            if i.timestamp >= recent_window
        ]
        
        # Group by target agent to find overloaded agents
        target_loads = defaultdict(list)
        for interaction in recent_interactions:
            target_loads[interaction.target_agent].append(interaction.duration_ms)
        
        # Identify bottlenecks
        bottlenecks = []
        for agent_id, durations in target_loads.items():
            if len(durations) > 10:  # High interaction volume
                avg_duration = statistics.mean(durations)
                if avg_duration > 5000:  # High average duration (5 seconds)
                    bottlenecks.append({
                        'agent_id': agent_id,
                        'interaction_count': len(durations),
                        'avg_duration_ms': avg_duration,
                        'bottleneck_type': 'high_load'
                    })
        
        # Update bottleneck detection
        self.bottleneck_detection[current_time.isoformat()] = bottlenecks
    
    def _analyze_collaboration_patterns(self):
        """Analyze collaboration patterns and efficiency"""
        
        # Analyze communication patterns
        communication_matrix = defaultdict(lambda: defaultdict(int))
        
        for interaction in self.interactions:
            communication_matrix[interaction.source_agent][interaction.target_agent] += 1
        
        # Calculate collaboration efficiency metrics
        for source_agent in communication_matrix:
            total_outgoing = sum(communication_matrix[source_agent].values())
            unique_targets = len(communication_matrix[source_agent])
            
            # Store pattern analysis
            self.collaboration_patterns[source_agent] = {
                'total_outgoing_interactions': total_outgoing,
                'unique_collaboration_partners': unique_targets,
                'collaboration_diversity': unique_targets / max(total_outgoing, 1),
                'most_frequent_partner': max(
                    communication_matrix[source_agent].items(),
                    key=lambda x: x[1],
                    default=(None, 0)
                )[0]
            }
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics with current data"""
        
        # Update agent state metrics
        for agent_id, state in self.agent_states.items():
            # Update based on latest metrics
            if agent_id in self.agent_metrics and self.agent_metrics[agent_id]:
                latest_metrics = self.agent_metrics[agent_id][-1]
                
                self.agent_memory_usage.labels(
                    agent_id=agent_id,
                    agent_type=latest_metrics.agent_type
                ).set(latest_metrics.memory_usage_mb * 1024 * 1024)
                
                self.agent_cpu_usage.labels(
                    agent_id=agent_id,
                    agent_type=latest_metrics.agent_type
                ).set(latest_metrics.cpu_usage_percent)
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean up old interactions
        while self.interactions and self.interactions[0].timestamp < cutoff_time:
            self.interactions.popleft()
        
        # Clean up old agent metrics
        for agent_id in self.agent_metrics:
            metrics_list = self.agent_metrics[agent_id]
            while metrics_list and metrics_list[0].timestamp < cutoff_time:
                metrics_list.popleft()
        
        # Clean up old workflow data
        workflows_to_remove = []
        for workflow_id, workflow in self.workflows.items():
            if workflow.end_time and workflow.end_time < cutoff_time:
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.workflows[workflow_id]
    
    def get_agent_performance_summary(
        self,
        agent_id: str,
        time_range_hours: int = 1
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary for an agent"""
        
        if agent_id not in self.agent_metrics:
            return {"error": "Agent not found"}
        
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        recent_metrics = [
            m for m in self.agent_metrics[agent_id]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics found"}
        
        # Calculate performance statistics
        processing_times = [m.processing_time_ms for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        
        summary = {
            "agent_id": agent_id,
            "agent_type": recent_metrics[0].agent_type,
            "time_range_hours": time_range_hours,
            "metrics_count": len(recent_metrics),
            "current_state": self.agent_states.get(agent_id, AgentState.IDLE).value,
            
            "performance": {
                "avg_processing_time_ms": statistics.mean(processing_times),
                "max_processing_time_ms": max(processing_times),
                "min_processing_time_ms": min(processing_times),
                "processing_time_std": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                
                "avg_memory_usage_mb": statistics.mean(memory_usage),
                "peak_memory_usage_mb": max(memory_usage),
                
                "avg_cpu_usage_percent": statistics.mean(cpu_usage),
                "peak_cpu_usage_percent": max(cpu_usage),
                
                "avg_success_rate": statistics.mean(success_rates),
                "min_success_rate": min(success_rates)
            },
            
            "ai_metrics": {
                "total_tokens_processed": sum(m.tokens_processed for m in recent_metrics),
                "total_api_calls": sum(m.api_calls_made for m in recent_metrics),
                "total_cost_usd": sum(m.cost_incurred_usd for m in recent_metrics),
                "avg_quality_score": statistics.mean([m.quality_score for m in recent_metrics])
            },
            
            "collaboration": {
                "total_messages_sent": sum(m.messages_sent for m in recent_metrics),
                "total_messages_received": sum(m.messages_received for m in recent_metrics),
                "collaborations_initiated": sum(m.collaborations_initiated for m in recent_metrics),
                "collaboration_completion_rate": (
                    sum(m.collaborations_completed for m in recent_metrics) /
                    max(sum(m.collaborations_initiated for m in recent_metrics), 1)
                )
            },
            
            "business_impact": {
                "tasks_completed": sum(m.tasks_completed for m in recent_metrics),
                "business_value_generated": sum(m.business_value_generated for m in recent_metrics),
                "avg_user_satisfaction": statistics.mean([m.user_satisfaction_score for m in recent_metrics])
            }
        }
        
        return summary
    
    def get_collaboration_network_analysis(self) -> Dict[str, Any]:
        """Analyze the collaboration network structure"""
        
        if not self.collaboration_graph.nodes():
            return {"error": "No collaboration data available"}
        
        # Calculate network metrics
        analysis = {
            "network_size": {
                "total_agents": self.collaboration_graph.number_of_nodes(),
                "total_interactions": self.collaboration_graph.number_of_edges(),
                "average_connections_per_agent": (
                    self.collaboration_graph.number_of_edges() / 
                    max(self.collaboration_graph.number_of_nodes(), 1)
                )
            },
            
            "centrality_metrics": {},
            "collaboration_clusters": [],
            "bottleneck_agents": [],
            "isolation_issues": []
        }
        
        # Calculate centrality metrics
        try:
            betweenness = nx.betweenness_centrality(self.collaboration_graph)
            closeness = nx.closeness_centrality(self.collaboration_graph)
            degree = nx.degree_centrality(self.collaboration_graph)
            
            # Find most central agents
            most_central_agents = {
                "betweenness": max(betweenness.items(), key=lambda x: x[1], default=(None, 0)),
                "closeness": max(closeness.items(), key=lambda x: x[1], default=(None, 0)),
                "degree": max(degree.items(), key=lambda x: x[1], default=(None, 0))
            }
            
            analysis["centrality_metrics"] = {
                "most_central_agents": most_central_agents,
                "betweenness_centrality": dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]),
                "closeness_centrality": dict(sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]),
                "degree_centrality": dict(sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            analysis["centrality_metrics"]["error"] = str(e)
        
        # Detect isolated agents
        isolated_agents = [node for node in self.collaboration_graph.nodes() 
                          if self.collaboration_graph.degree(node) == 0]
        analysis["isolation_issues"] = isolated_agents
        
        return analysis
    
    def generate_performance_report(
        self,
        time_range_hours: int = 24,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_range_hours": time_range_hours,
                "monitoring_period": f"{datetime.now() - timedelta(hours=time_range_hours)} to {datetime.now()}"
            },
            
            "system_overview": {
                "total_agents_monitored": len(self.agent_metrics),
                "active_workflows": len([w for w in self.workflows.values() if w.end_time is None]),
                "completed_workflows": len([w for w in self.workflows.values() if w.end_time is not None]),
                "total_interactions": len(self.interactions),
                "current_agent_states": dict(Counter([state.value for state in self.agent_states.values()]))
            },
            
            "performance_summary": {},
            "collaboration_analysis": self.get_collaboration_network_analysis(),
            "bottlenecks_detected": list(self.bottleneck_detection.values())[-10:],  # Last 10 detections
            "recommendations": []
        }
        
        # Generate per-agent summaries
        agent_summaries = {}
        for agent_id in self.agent_metrics.keys():
            agent_summaries[agent_id] = self.get_agent_performance_summary(agent_id, time_range_hours)
        
        report["agent_performance"] = agent_summaries
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Check for performance issues
        for agent_id, summary in agent_summaries.items():
            if isinstance(summary, dict) and "performance" in summary:
                perf = summary["performance"]
                
                if perf["avg_success_rate"] < 0.8:
                    recommendations.append({
                        "type": "performance_issue",
                        "agent": agent_id,
                        "issue": "Low success rate",
                        "details": f"Success rate of {perf['avg_success_rate']:.2%} is below threshold",
                        "suggestion": "Review error logs and consider model retraining or parameter tuning"
                    })
                
                if perf["avg_processing_time_ms"] > 10000:
                    recommendations.append({
                        "type": "latency_issue",
                        "agent": agent_id,
                        "issue": "High processing latency",
                        "details": f"Average processing time of {perf['avg_processing_time_ms']:.0f}ms is high",
                        "suggestion": "Consider optimizing agent logic or scaling resources"
                    })
        
        # Check for collaboration issues
        collab_analysis = report["collaboration_analysis"]
        if isinstance(collab_analysis, dict) and "isolation_issues" in collab_analysis:
            for isolated_agent in collab_analysis["isolation_issues"]:
                recommendations.append({
                    "type": "collaboration_issue",
                    "agent": isolated_agent,
                    "issue": "Agent isolation",
                    "details": "Agent has no collaboration connections",
                    "suggestion": "Review workflow design to ensure proper agent integration"
                })
        
        report["recommendations"] = recommendations
        
        return report
    
    def visualize_collaboration_network(self, save_path: Optional[str] = None) -> str:
        """Create visualization of the collaboration network"""
        
        if not self.collaboration_graph.nodes():
            print("No collaboration data to visualize")
            return ""
        
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.collaboration_graph, k=1, iterations=50)
        
        # Draw nodes with different colors based on agent state
        node_colors = []
        for node in self.collaboration_graph.nodes():
            state = self.agent_states.get(node, AgentState.IDLE)
            if state == AgentState.ACTIVE:
                node_colors.append('green')
            elif state == AgentState.ERROR:
                node_colors.append('red')
            elif state == AgentState.WAITING:
                node_colors.append('yellow')
            else:
                node_colors.append('lightblue')
        
        # Draw the network
        nx.draw_networkx_nodes(
            self.collaboration_graph, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8
        )
        
        nx.draw_networkx_labels(
            self.collaboration_graph, pos,
            font_size=8,
            font_weight='bold'
        )
        
        nx.draw_networkx_edges(
            self.collaboration_graph, pos,
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            edge_color='gray'
        )
        
        plt.title("Agent Collaboration Network", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Active'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Error'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Waiting'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Idle')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Collaboration network visualization saved to {save_path}")
        else:
            save_path = f"/tmp/collaboration_network_{int(time.time())}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path


# Example usage and testing
if __name__ == "__main__":
    # Initialize monitor
    monitor = AgentCollaborationMonitor()
    monitor.start_monitoring()
    
    # Simulate agent metrics
    ceo_metrics = AgentMetrics(
        agent_id="ceo-001",
        agent_type="ceo",
        timestamp=datetime.now(),
        processing_time_ms=1500,
        memory_usage_mb=256,
        cpu_usage_percent=45,
        success_rate=0.95,
        error_count=0,
        tokens_processed=1200,
        api_calls_made=5,
        cost_incurred_usd=0.024,
        quality_score=0.87,
        messages_sent=3,
        messages_received=2,
        collaborations_initiated=1,
        collaborations_completed=1,
        tasks_completed=1,
        business_value_generated=100.0,
        user_satisfaction_score=0.9
    )
    
    monitor.record_agent_metrics(ceo_metrics)
    
    # Simulate interaction
    interaction = AgentInteraction(
        interaction_id=str(uuid.uuid4()),
        source_agent="ceo-001",
        target_agent="cto-001",
        interaction_type=InteractionType.WORKFLOW_HANDOFF,
        timestamp=datetime.now(),
        duration_ms=250,
        success=True,
        payload_size_kb=5.2,
        metadata={"workflow_stage": "technical_validation"}
    )
    
    monitor.record_agent_interaction(interaction)
    
    # Start workflow monitoring
    workflow_id = monitor.start_workflow_monitoring(
        "wf-001",
        "startup_validation",
        ["ceo-001", "cto-001", "investor-001"]
    )
    
    # Generate performance report
    time.sleep(2)  # Allow some time for monitoring
    
    report = monitor.generate_performance_report(time_range_hours=1)
    print("Performance Report Generated:")
    print(json.dumps(report, indent=2, default=str))
    
    # Visualize collaboration network
    viz_path = monitor.visualize_collaboration_network()
    print(f"Network visualization saved to: {viz_path}")
    
    # Stop monitoring
    monitor.stop_monitoring()