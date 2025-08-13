"""
Quantum-Inspired Task Planning Module

This module implements quantum computing principles for advanced task scheduling,
resource optimization, and dependency management in the agentic startup studio.
"""

# Import optional dependencies with graceful fallback
try:
    from .quantum_dependencies import DependencyGraph, QuantumEntanglement
    QUANTUM_DEPENDENCIES_AVAILABLE = True
except ImportError:
    QUANTUM_DEPENDENCIES_AVAILABLE = False
    DependencyGraph = None
    QuantumEntanglement = None

try:
    from .quantum_planner import QuantumState, QuantumTask, QuantumTaskPlanner
    QUANTUM_PLANNER_AVAILABLE = True
except ImportError:
    QUANTUM_PLANNER_AVAILABLE = False
    QuantumState = None
    QuantumTask = None
    QuantumTaskPlanner = None

try:
    from .quantum_scheduler import QuantumScheduler, SuperpositionScheduler
    QUANTUM_SCHEDULER_AVAILABLE = True
except ImportError:
    QUANTUM_SCHEDULER_AVAILABLE = False
    QuantumScheduler = None
    SuperpositionScheduler = None

__all__ = [
    "QUANTUM_DEPENDENCIES_AVAILABLE",
    "QUANTUM_PLANNER_AVAILABLE", 
    "QUANTUM_SCHEDULER_AVAILABLE"
]

# Add optional components to __all__ if available
if QUANTUM_DEPENDENCIES_AVAILABLE:
    __all__.extend(["DependencyGraph", "QuantumEntanglement"])
    
if QUANTUM_PLANNER_AVAILABLE:
    __all__.extend(["QuantumTaskPlanner", "QuantumTask", "QuantumState"])
    
if QUANTUM_SCHEDULER_AVAILABLE:
    __all__.extend(["QuantumScheduler", "SuperpositionScheduler"])
