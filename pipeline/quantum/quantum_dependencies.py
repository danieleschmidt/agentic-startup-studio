"""
Quantum-Inspired Dependency Management

Implements quantum entanglement for advanced dependency tracking and resolution:
- Quantum entanglement: Correlated dependencies across the task system
- Non-local correlations: Dependencies that affect distant tasks instantaneously  
- Quantum superposition: Dependencies that exist in multiple states
- Bell inequality violations: Non-classical dependency relationships
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

import networkx as nx
import numpy as np

from .quantum_planner import QuantumState, QuantumTask

logger = logging.getLogger(__name__)


class EntanglementType(str, Enum):
    """Types of quantum entanglement between tasks."""
    SYNC_COMPLETION = "sync_completion"    # Tasks must complete together
    ANTI_CORRELATION = "anti_correlation"  # One fails if other succeeds
    PHASE_LOCK = "phase_lock"             # Tasks maintain phase relationship
    RESOURCE_SHARE = "resource_share"      # Tasks share quantum resources
    STATE_MIRROR = "state_mirror"         # Tasks mirror each other's states


@dataclass
class QuantumEntanglement:
    """
    Represents quantum entanglement between two or more tasks.
    
    Quantum entanglement in task planning allows for non-local correlations
    where the state of one task instantly affects entangled tasks regardless
    of distance in the dependency graph.
    """
    id: UUID
    task_ids: set[UUID]
    entanglement_type: EntanglementType
    strength: float = 1.0  # Entanglement strength (0-1)
    phase_correlation: float = 0.0  # Phase relationship
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

    def __post_init__(self):
        """Validate entanglement parameters."""
        if len(self.task_ids) < 2:
            raise ValueError("Entanglement requires at least 2 tasks")

        self.strength = max(0.0, min(1.0, self.strength))


class QuantumDependencyGraph:
    """
    Advanced dependency graph using quantum principles.
    
    Extends traditional dependency graphs with quantum properties:
    - Superposition edges (dependencies in multiple states)
    - Entangled nodes (non-local correlations)
    - Quantum interference in path calculations
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entanglements: dict[UUID, QuantumEntanglement] = {}
        self.quantum_correlations: dict[tuple[UUID, UUID], float] = {}
        self.measurement_history: list[dict[str, Any]] = []

    def add_task(self, task: QuantumTask) -> None:
        """Add a task to the quantum dependency graph."""
        self.graph.add_node(task.id, task=task)
        logger.debug(f"Added task {task.id} to quantum dependency graph")

    def add_dependency(self, dependent_id: UUID, dependency_id: UUID,
                      weight: float = 1.0, quantum_phase: float = 0.0) -> None:
        """
        Add a quantum dependency edge.
        
        Args:
            dependent_id: Task that depends on another
            dependency_id: Task that is depended upon
            weight: Dependency strength (0-1)
            quantum_phase: Quantum phase for interference calculations
        """
        if dependent_id not in self.graph or dependency_id not in self.graph:
            raise ValueError("Both tasks must be added to graph before creating dependency")

        self.graph.add_edge(
            dependency_id, dependent_id,
            weight=weight,
            quantum_phase=quantum_phase,
            created_at=datetime.utcnow()
        )

        logger.debug(f"Added quantum dependency: {dependency_id} -> {dependent_id}")

    def create_entanglement(self, task_ids: set[UUID],
                           entanglement_type: EntanglementType,
                           strength: float = 1.0) -> UUID:
        """
        Create quantum entanglement between tasks.
        
        Args:
            task_ids: Set of task IDs to entangle
            entanglement_type: Type of entanglement
            strength: Entanglement strength
            
        Returns:
            Entanglement ID
        """
        from uuid import uuid4

        # Validate all tasks exist
        for task_id in task_ids:
            if task_id not in self.graph:
                raise ValueError(f"Task {task_id} not found in graph")

        entanglement_id = uuid4()
        entanglement = QuantumEntanglement(
            id=entanglement_id,
            task_ids=task_ids,
            entanglement_type=entanglement_type,
            strength=strength
        )

        self.entanglements[entanglement_id] = entanglement

        # Update task entanglement references
        for task_id in task_ids:
            task = self.graph.nodes[task_id]['task']
            task.entangled_tasks.update(task_ids - {task_id})

        logger.info(f"Created {entanglement_type} entanglement: {entanglement_id}")
        return entanglement_id

    async def quantum_dependency_resolution(self, task_id: UUID) -> list[UUID]:
        """
        Resolve dependencies using quantum algorithms.
        
        Uses quantum interference to find optimal dependency paths
        and considers entanglement effects.
        
        Args:
            task_id: Task to resolve dependencies for
            
        Returns:
            List of dependency task IDs in optimal resolution order
        """
        if task_id not in self.graph:
            return []

        # Get all dependencies using quantum path analysis
        quantum_paths = await self._quantum_path_analysis(task_id)

        # Apply entanglement effects
        entangled_dependencies = self._resolve_entangled_dependencies(task_id)

        # Combine and optimize using quantum interference
        all_dependencies = set()
        for path in quantum_paths:
            all_dependencies.update(path)
        all_dependencies.update(entangled_dependencies)

        # Remove self-reference
        all_dependencies.discard(task_id)

        if not all_dependencies:
            return []

        # Quantum optimization of dependency order
        optimal_order = await self._quantum_optimize_dependency_order(
            list(all_dependencies), task_id
        )

        logger.info(f"Resolved {len(optimal_order)} dependencies for task {task_id}")
        return optimal_order

    async def _quantum_path_analysis(self, task_id: UUID) -> list[list[UUID]]:
        """
        Analyze all possible dependency paths using quantum superposition.
        
        Args:
            task_id: Target task
            
        Returns:
            List of possible dependency paths
        """
        paths = []

        # Find all paths to dependencies using NetworkX
        try:
            # Get all predecessors (dependencies)
            predecessors = list(self.graph.predecessors(task_id))

            for predecessor in predecessors:
                # Get paths from root nodes to this predecessor
                root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

                for root in root_nodes:
                    try:
                        if nx.has_path(self.graph, root, predecessor):
                            path = nx.shortest_path(self.graph, root, predecessor)
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

            # Direct dependencies path
            if predecessors:
                paths.append(predecessors)

        except Exception as e:
            logger.warning(f"Error in quantum path analysis: {e}")

        return paths

    def _resolve_entangled_dependencies(self, task_id: UUID) -> set[UUID]:
        """
        Resolve dependencies through quantum entanglement.
        
        Args:
            task_id: Task to check for entangled dependencies
            
        Returns:
            Set of entangled dependency task IDs
        """
        entangled_deps = set()

        # Check all entanglements involving this task
        for entanglement in self.entanglements.values():
            if not entanglement.active or task_id not in entanglement.task_ids:
                continue

            other_tasks = entanglement.task_ids - {task_id}

            # Apply entanglement type logic
            if entanglement.entanglement_type == EntanglementType.SYNC_COMPLETION:
                # All entangled tasks are dependencies of each other
                entangled_deps.update(other_tasks)

            elif entanglement.entanglement_type == EntanglementType.RESOURCE_SHARE:
                # Shared resource tasks may need coordination
                entangled_deps.update(other_tasks)

            elif entanglement.entanglement_type == EntanglementType.PHASE_LOCK:
                # Phase-locked tasks have temporal dependencies
                entangled_deps.update(other_tasks)

        return entangled_deps

    async def _quantum_optimize_dependency_order(self, dependencies: list[UUID],
                                               target_task_id: UUID) -> list[UUID]:
        """
        Optimize dependency execution order using quantum algorithms.
        
        Args:
            dependencies: List of dependency task IDs
            target_task_id: Task these dependencies are for
            
        Returns:
            Optimized dependency order
        """
        if not dependencies:
            return []

        # Create quantum amplitude matrix for dependencies
        n = len(dependencies)
        amplitude_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)

        # Apply quantum interference based on graph structure
        for i, dep1 in enumerate(dependencies):
            for j, dep2 in enumerate(dependencies):
                if i != j:
                    # Check if there's a direct dependency
                    if self.graph.has_edge(dep1, dep2):
                        edge_data = self.graph[dep1][dep2]
                        phase = edge_data.get('quantum_phase', 0.0)
                        weight = edge_data.get('weight', 1.0)

                        # Constructive interference for direct dependencies
                        amplitude_matrix[i, j] *= weight * np.exp(1j * phase)

                    # Check for entanglement correlations
                    correlation = self.quantum_correlations.get((dep1, dep2), 0.0)
                    if correlation != 0.0:
                        amplitude_matrix[i, j] *= (1.0 + correlation)

        # Quantum measurement to collapse to optimal order
        probabilities = np.abs(amplitude_matrix) ** 2

        # Use topological sort as base order, then apply quantum optimization
        subgraph = self.graph.subgraph(dependencies)
        try:
            topo_order = list(nx.topological_sort(subgraph))
        except nx.NetworkXError:
            # Handle cycles - use probabilistic ordering
            topo_order = dependencies.copy()

        # Apply quantum corrections to topological order
        quantum_order = []
        remaining = set(topo_order)

        while remaining:
            # Calculate quantum probabilities for next task
            next_probs = {}

            for task_id in remaining:
                task_idx = dependencies.index(task_id)

                # Base probability from quantum matrix
                prob = np.mean(probabilities[task_idx, :])

                # Boost probability if dependencies are satisfied
                deps_satisfied = True
                for pred in self.graph.predecessors(task_id):
                    if pred in remaining:
                        deps_satisfied = False
                        break

                if deps_satisfied:
                    prob *= 2.0

                next_probs[task_id] = prob

            # Quantum measurement - select next task
            if next_probs:
                total_prob = sum(next_probs.values())
                if total_prob > 0:
                    normalized_probs = {k: v/total_prob for k, v in next_probs.items()}

                    # Weighted random selection
                    rand_val = np.random.random()
                    cumulative = 0.0
                    selected = None

                    for task_id, prob in normalized_probs.items():
                        cumulative += prob
                        if rand_val <= cumulative:
                            selected = task_id
                            break

                    if selected:
                        quantum_order.append(selected)
                        remaining.remove(selected)
                    else:
                        # Fallback - take first available
                        selected = next(iter(remaining))
                        quantum_order.append(selected)
                        remaining.remove(selected)
                else:
                    # All probabilities zero - take any
                    selected = next(iter(remaining))
                    quantum_order.append(selected)
                    remaining.remove(selected)

        return quantum_order

    async def measure_entanglement_correlations(self) -> dict[str, float]:
        """
        Measure quantum correlations in the entangled task system.
        
        Returns:
            Dictionary of correlation measurements
        """
        correlations = {}

        for entanglement in self.entanglements.values():
            if not entanglement.active or len(entanglement.task_ids) < 2:
                continue

            task_ids = list(entanglement.task_ids)

            # Measure pairwise correlations
            for i in range(len(task_ids)):
                for j in range(i + 1, len(task_ids)):
                    task1_id, task2_id = task_ids[i], task_ids[j]

                    if task1_id in self.graph and task2_id in self.graph:
                        task1 = self.graph.nodes[task1_id]['task']
                        task2 = self.graph.nodes[task2_id]['task']

                        # Calculate correlation based on quantum states
                        correlation = self._calculate_quantum_correlation(task1, task2, entanglement)

                        correlation_key = f"{task1_id}_{task2_id}_{entanglement.entanglement_type}"
                        correlations[correlation_key] = correlation

                        # Store for future use
                        self.quantum_correlations[(task1_id, task2_id)] = correlation
                        self.quantum_correlations[(task2_id, task1_id)] = correlation

        return correlations

    def _calculate_quantum_correlation(self, task1: QuantumTask, task2: QuantumTask,
                                     entanglement: QuantumEntanglement) -> float:
        """
        Calculate quantum correlation between two entangled tasks.
        
        Args:
            task1: First task
            task2: Second task  
            entanglement: Entanglement relationship
            
        Returns:
            Correlation value (-1 to 1)
        """
        # Get quantum amplitudes for both tasks
        task1_phases = [amp.phase for amp in task1.amplitudes.values()]
        task2_phases = [amp.phase for amp in task2.amplitudes.values()]

        if not task1_phases or not task2_phases:
            return 0.0

        # Calculate phase correlation
        avg_phase1 = np.mean(task1_phases)
        avg_phase2 = np.mean(task2_phases)
        phase_diff = abs(avg_phase1 - avg_phase2)

        # Base correlation from phase relationship
        base_correlation = np.cos(phase_diff) * entanglement.strength

        # Apply entanglement type modifiers
        if entanglement.entanglement_type == EntanglementType.ANTI_CORRELATION:
            base_correlation *= -1
        elif entanglement.entanglement_type == EntanglementType.SYNC_COMPLETION:
            base_correlation = abs(base_correlation)

        return np.clip(base_correlation, -1.0, 1.0)

    async def break_entanglement(self, entanglement_id: UUID) -> bool:
        """
        Break quantum entanglement between tasks.
        
        Args:
            entanglement_id: ID of entanglement to break
            
        Returns:
            True if entanglement was broken
        """
        if entanglement_id not in self.entanglements:
            return False

        entanglement = self.entanglements[entanglement_id]

        # Remove entanglement references from tasks
        for task_id in entanglement.task_ids:
            if task_id in self.graph:
                task = self.graph.nodes[task_id]['task']
                for other_id in entanglement.task_ids:
                    if other_id != task_id:
                        task.entangled_tasks.discard(other_id)

        # Remove quantum correlations
        task_list = list(entanglement.task_ids)
        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                self.quantum_correlations.pop((task_list[i], task_list[j]), None)
                self.quantum_correlations.pop((task_list[j], task_list[i]), None)

        # Remove entanglement
        del self.entanglements[entanglement_id]

        logger.info(f"Broke quantum entanglement: {entanglement_id}")
        return True

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the quantum dependency graph."""
        return {
            "total_tasks": len(self.graph.nodes),
            "total_dependencies": len(self.graph.edges),
            "total_entanglements": len(self.entanglements),
            "active_entanglements": sum(1 for e in self.entanglements.values() if e.active),
            "quantum_correlations": len(self.quantum_correlations),
            "graph_density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "measurement_history_length": len(self.measurement_history)
        }


class DependencyGraph:
    """
    High-level interface for quantum dependency management.
    
    Provides simplified API for common dependency operations while
    leveraging quantum algorithms underneath.
    """

    def __init__(self):
        self.quantum_graph = QuantumDependencyGraph()
        self.task_registry: dict[UUID, QuantumTask] = {}

    async def register_task(self, task: QuantumTask) -> None:
        """Register a task in the dependency system."""
        self.task_registry[task.id] = task
        self.quantum_graph.add_task(task)

    async def add_dependency(self, dependent_task_id: UUID,
                           dependency_task_id: UUID) -> None:
        """Add a simple dependency relationship."""
        self.quantum_graph.add_dependency(dependent_task_id, dependency_task_id)

    async def create_task_group(self, task_ids: list[UUID],
                              sync_completion: bool = True) -> UUID:
        """
        Create a group of tasks with quantum entanglement.
        
        Args:
            task_ids: Tasks to group together
            sync_completion: Whether tasks should complete synchronously
            
        Returns:
            Entanglement ID for the group
        """
        entanglement_type = (EntanglementType.SYNC_COMPLETION if sync_completion
                           else EntanglementType.RESOURCE_SHARE)

        return self.quantum_graph.create_entanglement(
            set(task_ids), entanglement_type
        )

    async def get_ready_tasks(self) -> list[QuantumTask]:
        """
        Get all tasks that are ready to execute (dependencies satisfied).
        
        Returns:
            List of ready tasks
        """
        ready_tasks = []

        for task_id, task in self.task_registry.items():
            if task.current_state not in [QuantumState.PENDING, QuantumState.SUPERPOSITION]:
                continue

            # Check if dependencies are satisfied
            dependencies = await self.quantum_graph.quantum_dependency_resolution(task_id)

            deps_satisfied = True
            for dep_id in dependencies:
                if dep_id in self.task_registry:
                    dep_task = self.task_registry[dep_id]
                    if dep_task.current_state != QuantumState.COMPLETED:
                        deps_satisfied = False
                        break

            if deps_satisfied:
                ready_tasks.append(task)

        return ready_tasks

    async def get_dependency_chain(self, task_id: UUID) -> list[UUID]:
        """Get the full dependency chain for a task."""
        return await self.quantum_graph.quantum_dependency_resolution(task_id)

    async def analyze_system(self) -> dict[str, Any]:
        """Analyze the entire dependency system."""
        correlations = await self.quantum_graph.measure_entanglement_correlations()
        stats = self.quantum_graph.get_graph_statistics()

        return {
            "graph_statistics": stats,
            "quantum_correlations": correlations,
            "total_registered_tasks": len(self.task_registry),
            "ready_tasks": len(await self.get_ready_tasks())
        }
