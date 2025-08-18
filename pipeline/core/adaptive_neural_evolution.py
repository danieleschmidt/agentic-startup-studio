"""
Adaptive Neural Evolution Engine - Self-Improving AI Architecture

Implements evolutionary algorithms for:
- Dynamic neural architecture optimization
- Adaptive hyperparameter tuning
- Multi-objective performance optimization
- Continuous learning and adaptation
"""

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


class EvolutionStrategy(str, Enum):
    """Neural evolution strategies."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    NEUROEVOLUTION = "neuroevolution"
    HYBRID_APPROACH = "hybrid_approach"


class FitnessMetric(str, Enum):
    """Fitness evaluation metrics."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class NeuralGenome:
    """Represents a neural network genome for evolution."""
    genome_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Architecture parameters
    layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation_functions: List[str] = field(default_factory=lambda: ["relu", "relu", "sigmoid"])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.0])
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Evolution metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_count: int = 0
    
    # Performance tracking
    fitness_scores: Dict[FitnessMetric, float] = field(default_factory=dict)
    training_history: List[Dict[str, float]] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    
    # Adaptive parameters
    adaptation_rate: float = 0.1
    complexity_penalty: float = 0.01
    age: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)
    last_evaluated: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize genome with random values if needed."""
        if not self.fitness_scores:
            self.fitness_scores = {metric: 0.0 for metric in FitnessMetric}
    
    def calculate_complexity(self) -> float:
        """Calculate genome complexity score."""
        total_params = sum(
            self.layers[i] * self.layers[i+1] 
            for i in range(len(self.layers) - 1)
        )
        return total_params / 10000.0  # Normalized complexity
    
    def get_multi_objective_fitness(self) -> float:
        """Calculate multi-objective fitness score."""
        accuracy_weight = 0.4
        latency_weight = 0.3
        efficiency_weight = 0.3
        
        # Normalize and combine metrics
        accuracy = self.fitness_scores.get(FitnessMetric.ACCURACY, 0.0)
        latency = 1.0 - min(1.0, self.fitness_scores.get(FitnessMetric.LATENCY, 1.0))  # Invert latency
        efficiency = self.fitness_scores.get(FitnessMetric.RESOURCE_EFFICIENCY, 0.0)
        
        return (accuracy * accuracy_weight + 
                latency * latency_weight + 
                efficiency * efficiency_weight)


@dataclass
class EvolutionParameters:
    """Parameters controlling the evolution process."""
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Selection parameters
    tournament_size: int = 5
    selection_pressure: float = 1.5
    
    # Mutation parameters
    mutation_strength: float = 0.1
    adaptive_mutation: bool = True
    
    # Diversity maintenance
    diversity_threshold: float = 0.1
    novelty_search: bool = True
    
    # Termination criteria
    max_generations: int = 100
    target_fitness: float = 0.95
    convergence_patience: int = 10


class AdaptiveNeuralEvolution:
    """
    Advanced neural evolution engine with adaptive capabilities.
    
    Features:
    - Multi-strategy evolution algorithms
    - Dynamic population management
    - Adaptive hyperparameter optimization
    - Continuous learning and improvement
    """
    
    def __init__(self, 
                 evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID_APPROACH,
                 fitness_metric: FitnessMetric = FitnessMetric.MULTI_OBJECTIVE,
                 parameters: Optional[EvolutionParameters] = None):
        """Initialize the adaptive neural evolution engine."""
        
        self.evolution_strategy = evolution_strategy
        self.fitness_metric = fitness_metric
        self.parameters = parameters or EvolutionParameters()
        
        # Population management
        self.population: List[NeuralGenome] = []
        self.elite_population: List[NeuralGenome] = []
        self.archive: List[NeuralGenome] = []
        
        # Evolution tracking
        self.current_generation = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.convergence_counter = 0
        
        # Adaptive mechanisms
        self.adaptive_params = {
            "mutation_rate": self.parameters.mutation_rate,
            "crossover_rate": self.parameters.crossover_rate,
            "selection_pressure": self.parameters.selection_pressure
        }
        
        # Performance tracking
        self.evaluation_times: List[float] = []
        self.generation_times: List[float] = []
        
        logger.info(f"Adaptive Neural Evolution initialized with {evolution_strategy.value} strategy")
    
    def initialize_population(self, custom_genomes: Optional[List[NeuralGenome]] = None) -> None:
        """Initialize the evolution population."""
        if custom_genomes:
            self.population = custom_genomes[:self.parameters.population_size]
        else:
            self.population = []
        
        # Fill remaining slots with random genomes
        while len(self.population) < self.parameters.population_size:
            genome = self._create_random_genome()
            self.population.append(genome)
        
        logger.info(f"Initialized population with {len(self.population)} genomes")
    
    async def evolve_population(self, 
                               fitness_evaluator: callable,
                               max_generations: Optional[int] = None) -> NeuralGenome:
        """
        Evolve the population to optimize neural architectures.
        
        Args:
            fitness_evaluator: Function to evaluate genome fitness
            max_generations: Override default max generations
            
        Returns:
            NeuralGenome: Best evolved genome
        """
        max_gens = max_generations or self.parameters.max_generations
        
        logger.info(f"Starting evolution for {max_gens} generations")
        start_time = time.time()
        
        try:
            for generation in range(max_gens):
                gen_start_time = time.time()
                self.current_generation = generation
                
                # Evaluate population fitness
                await self._evaluate_population(fitness_evaluator)
                
                # Update elite population
                self._update_elite_population()
                
                # Check termination criteria
                if self._should_terminate():
                    logger.info(f"Evolution terminated early at generation {generation}")
                    break
                
                # Generate next generation
                await self._generate_next_generation()
                
                # Adaptive parameter adjustment
                self._adapt_evolution_parameters()
                
                # Record generation metrics
                gen_time = time.time() - gen_start_time
                self.generation_times.append(gen_time)
                
                # Log progress
                best_fitness = max(g.get_multi_objective_fitness() for g in self.population)
                self.best_fitness_history.append(best_fitness)
                
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            total_time = time.time() - start_time
            best_genome = self._get_best_genome()
            
            logger.info(f"Evolution completed in {total_time:.2f}s")
            logger.info(f"Best fitness achieved: {best_genome.get_multi_objective_fitness():.4f}")
            
            return best_genome
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise
    
    async def _evaluate_population(self, fitness_evaluator: callable):
        """Evaluate fitness for all genomes in the population."""
        evaluation_tasks = []
        
        for genome in self.population:
            # Skip recently evaluated genomes if using caching
            if (genome.last_evaluated and 
                datetime.now() - genome.last_evaluated < timedelta(minutes=5)):
                continue
            
            task = self._evaluate_genome_fitness(genome, fitness_evaluator)
            evaluation_tasks.append(task)
        
        if evaluation_tasks:
            await asyncio.gather(*evaluation_tasks, return_exceptions=True)
    
    async def _evaluate_genome_fitness(self, genome: NeuralGenome, fitness_evaluator: callable):
        """Evaluate fitness for a single genome."""
        try:
            start_time = time.time()
            
            # Call the fitness evaluator
            if asyncio.iscoroutinefunction(fitness_evaluator):
                fitness_results = await fitness_evaluator(genome)
            else:
                fitness_results = await asyncio.get_event_loop().run_in_executor(
                    None, fitness_evaluator, genome
                )
            
            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            
            # Update genome fitness scores
            if isinstance(fitness_results, dict):
                for metric, score in fitness_results.items():
                    if isinstance(metric, str):
                        metric = FitnessMetric(metric)
                    genome.fitness_scores[metric] = score
            else:
                # Single fitness value
                genome.fitness_scores[self.fitness_metric] = fitness_results
            
            genome.last_evaluated = datetime.now()
            genome.age += 1
            
        except Exception as e:
            logger.error(f"Failed to evaluate genome {genome.genome_id}: {e}")
            # Assign poor fitness on evaluation failure
            for metric in FitnessMetric:
                genome.fitness_scores[metric] = 0.0
    
    def _update_elite_population(self):
        """Update the elite population with best performing genomes."""
        # Sort population by multi-objective fitness
        sorted_population = sorted(
            self.population,
            key=lambda g: g.get_multi_objective_fitness(),
            reverse=True
        )
        
        # Update elite population
        self.elite_population = sorted_population[:self.parameters.elite_size]
        
        # Archive exceptional genomes
        for genome in self.elite_population:
            if genome.get_multi_objective_fitness() > 0.9:  # High-performance threshold
                if genome not in self.archive:
                    self.archive.append(genome)
    
    async def _generate_next_generation(self):
        """Generate the next generation using evolution strategies."""
        new_population = []
        
        # Always keep elite genomes
        new_population.extend(self.elite_population.copy())
        
        # Generate offspring based on evolution strategy
        while len(new_population) < self.parameters.population_size:
            if self.evolution_strategy == EvolutionStrategy.GENETIC_ALGORITHM:
                offspring = await self._genetic_algorithm_step()
            elif self.evolution_strategy == EvolutionStrategy.DIFFERENTIAL_EVOLUTION:
                offspring = await self._differential_evolution_step()
            elif self.evolution_strategy == EvolutionStrategy.PARTICLE_SWARM:
                offspring = await self._particle_swarm_step()
            elif self.evolution_strategy == EvolutionStrategy.NEUROEVOLUTION:
                offspring = await self._neuroevolution_step()
            else:  # HYBRID_APPROACH
                offspring = await self._hybrid_evolution_step()
            
            if offspring:
                new_population.append(offspring)
        
        # Update population
        self.population = new_population[:self.parameters.population_size]
        
        # Increment generation for all genomes
        for genome in self.population:
            genome.generation = self.current_generation + 1
    
    async def _genetic_algorithm_step(self) -> Optional[NeuralGenome]:
        """Generate offspring using genetic algorithm."""
        # Tournament selection
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        # Crossover
        if random.random() < self.adaptive_params["crossover_rate"]:
            offspring = self._crossover(parent1, parent2)
        else:
            offspring = parent1
        
        # Mutation
        if random.random() < self.adaptive_params["mutation_rate"]:
            offspring = self._mutate(offspring)
        
        return offspring
    
    async def _differential_evolution_step(self) -> Optional[NeuralGenome]:
        """Generate offspring using differential evolution."""
        # Select three random genomes
        candidates = random.sample(self.population, 3)
        base, diff1, diff2 = candidates
        
        # Create mutant vector
        mutant = self._differential_mutation(base, diff1, diff2)
        
        # Crossover with original
        target = random.choice(self.population)
        offspring = self._differential_crossover(target, mutant)
        
        return offspring
    
    async def _particle_swarm_step(self) -> Optional[NeuralGenome]:
        """Generate offspring using particle swarm optimization principles."""
        # Select particle (genome) and update based on best positions
        particle = random.choice(self.population)
        global_best = self._get_best_genome()
        
        # Create new position based on particle swarm dynamics
        offspring = self._particle_swarm_update(particle, global_best)
        
        return offspring
    
    async def _neuroevolution_step(self) -> Optional[NeuralGenome]:
        """Generate offspring using neuroevolution techniques."""
        # Select parent based on fitness
        parent = self._fitness_proportionate_selection()
        
        # Apply neuroevolution-specific mutations
        offspring = self._neuroevolution_mutate(parent)
        
        return offspring
    
    async def _hybrid_evolution_step(self) -> Optional[NeuralGenome]:
        """Generate offspring using hybrid approach."""
        # Randomly choose evolution strategy for this step
        strategies = [
            self._genetic_algorithm_step,
            self._differential_evolution_step,
            self._particle_swarm_step,
            self._neuroevolution_step
        ]
        
        chosen_strategy = random.choice(strategies)
        return await chosen_strategy()
    
    def _tournament_selection(self) -> NeuralGenome:
        """Select genome using tournament selection."""
        tournament = random.sample(self.population, self.parameters.tournament_size)
        return max(tournament, key=lambda g: g.get_multi_objective_fitness())
    
    def _fitness_proportionate_selection(self) -> NeuralGenome:
        """Select genome using fitness proportionate selection."""
        fitness_values = [g.get_multi_objective_fitness() for g in self.population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        # Roulette wheel selection
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for i, fitness in enumerate(fitness_values):
            current_sum += fitness
            if current_sum >= selection_point:
                return self.population[i]
        
        return self.population[-1]  # Fallback
    
    def _crossover(self, parent1: NeuralGenome, parent2: NeuralGenome) -> NeuralGenome:
        """Create offspring through crossover."""
        offspring = NeuralGenome()
        offspring.parent_ids = [parent1.genome_id, parent2.genome_id]
        
        # Blend architecture parameters
        offspring.layers = self._blend_lists(parent1.layers, parent2.layers)
        offspring.activation_functions = random.choice([parent1.activation_functions, parent2.activation_functions])
        offspring.dropout_rates = self._blend_lists(parent1.dropout_rates, parent2.dropout_rates)
        
        # Blend numeric parameters
        offspring.learning_rate = (parent1.learning_rate + parent2.learning_rate) / 2
        offspring.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
        
        return offspring
    
    def _mutate(self, genome: NeuralGenome) -> NeuralGenome:
        """Apply mutations to a genome."""
        mutant = NeuralGenome()
        mutant.genome_id = genome.genome_id
        mutant.parent_ids = genome.parent_ids.copy()
        mutant.mutation_count = genome.mutation_count + 1
        
        # Copy base parameters
        mutant.layers = genome.layers.copy()
        mutant.activation_functions = genome.activation_functions.copy()
        mutant.dropout_rates = genome.dropout_rates.copy()
        mutant.learning_rate = genome.learning_rate
        mutant.batch_size = genome.batch_size
        
        # Apply various mutations
        mutation_strength = self.parameters.mutation_strength
        
        # Layer size mutations
        if random.random() < 0.3:
            layer_idx = random.randint(0, len(mutant.layers) - 1)
            delta = int(random.gauss(0, mutation_strength * 50))
            mutant.layers[layer_idx] = max(1, mutant.layers[layer_idx] + delta)
        
        # Learning rate mutation
        if random.random() < 0.2:
            delta = random.gauss(0, mutation_strength * 0.001)
            mutant.learning_rate = max(0.0001, min(0.1, mutant.learning_rate + delta))
        
        # Dropout rate mutation
        if random.random() < 0.2:
            dropout_idx = random.randint(0, len(mutant.dropout_rates) - 1)
            delta = random.gauss(0, mutation_strength * 0.1)
            mutant.dropout_rates[dropout_idx] = max(0.0, min(0.5, mutant.dropout_rates[dropout_idx] + delta))
        
        # Activation function mutation
        if random.random() < 0.1:
            activations = ["relu", "tanh", "sigmoid", "leaky_relu", "elu"]
            func_idx = random.randint(0, len(mutant.activation_functions) - 1)
            mutant.activation_functions[func_idx] = random.choice(activations)
        
        return mutant
    
    def _differential_mutation(self, base: NeuralGenome, diff1: NeuralGenome, diff2: NeuralGenome) -> NeuralGenome:
        """Apply differential evolution mutation."""
        mutant = NeuralGenome()
        
        # Differential mutation: base + F * (diff1 - diff2)
        F = 0.5  # Differential weight
        
        # Blend numeric parameters using differential evolution
        mutant.learning_rate = base.learning_rate + F * (diff1.learning_rate - diff2.learning_rate)
        mutant.learning_rate = max(0.0001, min(0.1, mutant.learning_rate))
        
        # For discrete parameters, use probabilistic selection
        mutant.layers = base.layers.copy()
        mutant.activation_functions = random.choice([base.activation_functions, diff1.activation_functions, diff2.activation_functions])
        mutant.dropout_rates = base.dropout_rates.copy()
        mutant.batch_size = random.choice([base.batch_size, diff1.batch_size, diff2.batch_size])
        
        return mutant
    
    def _differential_crossover(self, target: NeuralGenome, mutant: NeuralGenome) -> NeuralGenome:
        """Apply differential evolution crossover."""
        offspring = NeuralGenome()
        CR = 0.7  # Crossover probability
        
        # Crossover parameters
        if random.random() < CR:
            offspring.learning_rate = mutant.learning_rate
        else:
            offspring.learning_rate = target.learning_rate
        
        offspring.layers = mutant.layers if random.random() < CR else target.layers
        offspring.activation_functions = mutant.activation_functions if random.random() < CR else target.activation_functions
        offspring.dropout_rates = mutant.dropout_rates if random.random() < CR else target.dropout_rates
        offspring.batch_size = mutant.batch_size if random.random() < CR else target.batch_size
        
        return offspring
    
    def _particle_swarm_update(self, particle: NeuralGenome, global_best: NeuralGenome) -> NeuralGenome:
        """Update particle position using PSO dynamics."""
        offspring = NeuralGenome()
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Update learning rate using PSO velocity
        velocity = w * 0.001 + c1 * random.random() * (particle.learning_rate - particle.learning_rate) + \
                  c2 * random.random() * (global_best.learning_rate - particle.learning_rate)
        
        offspring.learning_rate = max(0.0001, min(0.1, particle.learning_rate + velocity))
        
        # For discrete parameters, use probabilistic updates
        offspring.layers = particle.layers.copy()
        offspring.activation_functions = particle.activation_functions.copy()
        offspring.dropout_rates = particle.dropout_rates.copy()
        offspring.batch_size = particle.batch_size
        
        # Apply some randomization
        if random.random() < 0.1:
            offspring = self._mutate(offspring)
        
        return offspring
    
    def _neuroevolution_mutate(self, parent: NeuralGenome) -> NeuralGenome:
        """Apply neuroevolution-specific mutations."""
        offspring = self._mutate(parent)
        
        # Additional neuroevolution mutations
        # Add/remove layers
        if random.random() < 0.1:
            if len(offspring.layers) > 2 and random.random() < 0.5:
                # Remove layer
                remove_idx = random.randint(1, len(offspring.layers) - 2)
                offspring.layers.pop(remove_idx)
                offspring.activation_functions.pop(remove_idx)
                offspring.dropout_rates.pop(remove_idx)
            else:
                # Add layer
                add_idx = random.randint(1, len(offspring.layers) - 1)
                new_size = random.randint(16, 256)
                offspring.layers.insert(add_idx, new_size)
                offspring.activation_functions.insert(add_idx, "relu")
                offspring.dropout_rates.insert(add_idx, 0.1)
        
        return offspring
    
    def _blend_lists(self, list1: List, list2: List) -> List:
        """Blend two lists by randomly selecting elements."""
        min_len = min(len(list1), len(list2))
        result = []
        
        for i in range(min_len):
            result.append(random.choice([list1[i], list2[i]]))
        
        # Handle remaining elements
        if len(list1) > min_len:
            result.extend(list1[min_len:])
        elif len(list2) > min_len:
            result.extend(list2[min_len:])
        
        return result
    
    def _create_random_genome(self) -> NeuralGenome:
        """Create a random neural genome."""
        # Random architecture
        num_layers = random.randint(2, 6)
        layers = [random.randint(32, 512) for _ in range(num_layers)]
        
        activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
        activation_functions = [random.choice(activations) for _ in range(num_layers)]
        
        dropout_rates = [random.uniform(0.0, 0.3) for _ in range(num_layers)]
        
        return NeuralGenome(
            layers=layers,
            activation_functions=activation_functions,
            dropout_rates=dropout_rates,
            learning_rate=random.uniform(0.0001, 0.01),
            batch_size=random.choice([16, 32, 64, 128])
        )
    
    def _should_terminate(self) -> bool:
        """Check if evolution should terminate."""
        if not self.best_fitness_history:
            return False
        
        best_fitness = max(self.best_fitness_history)
        
        # Target fitness reached
        if best_fitness >= self.parameters.target_fitness:
            return True
        
        # Convergence check
        if len(self.best_fitness_history) >= self.parameters.convergence_patience:
            recent_fitness = self.best_fitness_history[-self.parameters.convergence_patience:]
            fitness_variance = np.var(recent_fitness)
            
            if fitness_variance < 0.001:  # Very low variance indicates convergence
                self.convergence_counter += 1
                if self.convergence_counter >= 3:
                    return True
            else:
                self.convergence_counter = 0
        
        return False
    
    def _adapt_evolution_parameters(self):
        """Adapt evolution parameters based on current performance."""
        if not self.best_fitness_history or len(self.best_fitness_history) < 5:
            return
        
        # Calculate recent improvement rate
        recent_fitness = self.best_fitness_history[-5:]
        improvement_rate = (recent_fitness[-1] - recent_fitness[0]) / 5
        
        # Adapt mutation rate
        if improvement_rate < 0.001:  # Low improvement
            self.adaptive_params["mutation_rate"] = min(0.3, self.adaptive_params["mutation_rate"] * 1.1)
        else:  # Good improvement
            self.adaptive_params["mutation_rate"] = max(0.05, self.adaptive_params["mutation_rate"] * 0.95)
        
        # Adapt crossover rate
        diversity = self._calculate_population_diversity()
        if diversity < self.parameters.diversity_threshold:
            self.adaptive_params["crossover_rate"] = min(0.9, self.adaptive_params["crossover_rate"] * 1.05)
        
        logger.debug(f"Adapted parameters: mutation_rate={self.adaptive_params['mutation_rate']:.3f}, "
                    f"crossover_rate={self.adaptive_params['crossover_rate']:.3f}")
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 1.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._genome_distance(self.population[i], self.population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _genome_distance(self, genome1: NeuralGenome, genome2: NeuralGenome) -> float:
        """Calculate distance between two genomes."""
        # Simple distance metric based on parameter differences
        distance = 0.0
        
        # Learning rate difference
        distance += abs(genome1.learning_rate - genome2.learning_rate) * 100
        
        # Batch size difference
        distance += abs(genome1.batch_size - genome2.batch_size) / 128.0
        
        # Layer architecture difference
        min_layers = min(len(genome1.layers), len(genome2.layers))
        for i in range(min_layers):
            distance += abs(genome1.layers[i] - genome2.layers[i]) / 512.0
        
        # Architecture length difference
        distance += abs(len(genome1.layers) - len(genome2.layers))
        
        return distance
    
    def _get_best_genome(self) -> NeuralGenome:
        """Get the best genome from current population."""
        return max(self.population, key=lambda g: g.get_multi_objective_fitness())
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        best_genome = self._get_best_genome()
        
        return {
            "current_generation": self.current_generation,
            "population_size": len(self.population),
            "elite_size": len(self.elite_population),
            "archive_size": len(self.archive),
            "best_fitness": best_genome.get_multi_objective_fitness(),
            "fitness_history": self.best_fitness_history,
            "diversity_history": self.diversity_history,
            "convergence_counter": self.convergence_counter,
            "adaptive_parameters": self.adaptive_params,
            "evaluation_times": {
                "mean": np.mean(self.evaluation_times) if self.evaluation_times else 0,
                "std": np.std(self.evaluation_times) if self.evaluation_times else 0
            },
            "generation_times": {
                "mean": np.mean(self.generation_times) if self.generation_times else 0,
                "std": np.std(self.generation_times) if self.generation_times else 0
            },
            "best_genome": {
                "genome_id": best_genome.genome_id,
                "layers": best_genome.layers,
                "learning_rate": best_genome.learning_rate,
                "batch_size": best_genome.batch_size,
                "complexity": best_genome.calculate_complexity(),
                "age": best_genome.age
            }
        }


# Factory function
def create_adaptive_neural_evolution(
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID_APPROACH,
    fitness_metric: FitnessMetric = FitnessMetric.MULTI_OBJECTIVE,
    parameters: Optional[EvolutionParameters] = None
) -> AdaptiveNeuralEvolution:
    """Create and return a configured Adaptive Neural Evolution engine."""
    return AdaptiveNeuralEvolution(
        evolution_strategy=evolution_strategy,
        fitness_metric=fitness_metric,
        parameters=parameters
    )


# Demonstration function
async def neural_evolution_demo():
    """Demonstrate adaptive neural evolution capabilities."""
    
    async def mock_fitness_evaluator(genome: NeuralGenome) -> Dict[FitnessMetric, float]:
        """Mock fitness evaluator for demonstration."""
        # Simulate evaluation time
        await asyncio.sleep(0.1)
        
        # Mock fitness based on architecture
        complexity = genome.calculate_complexity()
        
        # Simulate trade-offs
        accuracy = random.uniform(0.7, 0.95) - complexity * 0.05
        latency = random.uniform(0.1, 2.0) + complexity * 0.1
        efficiency = 1.0 - complexity * 0.2
        
        return {
            FitnessMetric.ACCURACY: max(0, accuracy),
            FitnessMetric.LATENCY: latency,
            FitnessMetric.RESOURCE_EFFICIENCY: max(0, efficiency)
        }
    
    # Create evolution engine
    evolution_params = EvolutionParameters(
        population_size=20,
        max_generations=15,
        target_fitness=0.9
    )
    
    engine = create_adaptive_neural_evolution(
        evolution_strategy=EvolutionStrategy.HYBRID_APPROACH,
        parameters=evolution_params
    )
    
    # Initialize population
    engine.initialize_population()
    
    # Run evolution
    best_genome = await engine.evolve_population(mock_fitness_evaluator)
    
    # Display results
    stats = engine.get_evolution_statistics()
    
    print("Adaptive Neural Evolution Results:")
    print(f"Best Fitness: {stats['best_fitness']:.4f}")
    print(f"Generations: {stats['current_generation']}")
    print(f"Best Architecture: {stats['best_genome']['layers']}")
    print(f"Learning Rate: {stats['best_genome']['learning_rate']:.6f}")
    print(f"Complexity: {stats['best_genome']['complexity']:.4f}")
    
    return best_genome, stats


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(neural_evolution_demo())