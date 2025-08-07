"""
Quantum-Inspired Task Planning Module

This module implements quantum computing principles for advanced task scheduling,
resource optimization, and dependency management in the agentic startup studio.
"""

from .quantum_dependencies import DependencyGraph, QuantumEntanglement
from .quantum_planner import QuantumState, QuantumTask, QuantumTaskPlanner
from .quantum_scheduler import QuantumScheduler, SuperpositionScheduler

__all__ = [
    "QuantumTaskPlanner",
    "QuantumTask",
    "QuantumState",
    "QuantumScheduler",
    "SuperpositionScheduler",
    "QuantumEntanglement",
    "DependencyGraph"
]
