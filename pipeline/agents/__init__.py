"""
Agents package - Multi-agent coordination and workflow management.

This package provides the agent orchestrator and specialized AI agents
for coordinating the startup studio pipeline workflow.
"""

from .agent_orchestrator import (
    AgentRole,
    WorkflowState, 
    AgentCapability,
    AgentContext,
    AgentDecision,
    BaseAgent,
    CEOAgent,
    CTOAgent,
    VCAgent,
    AgentOrchestrator,
    get_agent_orchestrator
)

__all__ = [
    'AgentRole',
    'WorkflowState',
    'AgentCapability', 
    'AgentContext',
    'AgentDecision',
    'BaseAgent',
    'CEOAgent',
    'CTOAgent',
    'VCAgent',
    'AgentOrchestrator',
    'get_agent_orchestrator'
]