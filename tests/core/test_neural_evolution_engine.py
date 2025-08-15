"""
Test suite for Neural Evolution Engine - Generation 1 Enhancement
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from pipeline.core.neural_evolution_engine import (
    NeuralEvolutionEngine,
    NeuralGenome,
    QuantumNeuron,
    QuantumNeuralNetwork,
    NeuralNetworkType,
    EvolutionStrategy,
    get_evolution_engine
)


class TestQuantumNeuron:
    """Test quantum neuron functionality"""
    
    def test_quantum_neuron_initialization(self):
        """Test quantum neuron creation"""
        neuron = QuantumNeuron(
            neuron_id="test_neuron",
            quantum_state=complex(0.5, 0.5)
        )
        
        assert neuron.neuron_id == "test_neuron"
        assert isinstance(neuron.quantum_state, complex)
        assert neuron.coherence_time == 1.0
        assert neuron.decoherence_rate == 0.01
        assert len(neuron.entanglement_partners) == 0
    
    def test_quantum_activation(self):
        """Test quantum activation function"""
        neuron = QuantumNeuron("test", complex(1, 0))
        inputs = np.array([0.5, 0.3, 0.8])
        
        result = neuron.quantum_activation(inputs)
        
        assert isinstance(result, complex)
        assert abs(result) <= 1.0  # Quantum state magnitude should be normalized
    
    def test_entanglement(self):
        """Test quantum entanglement between neurons"""
        neuron1 = QuantumNeuron("neuron1", complex(1, 0))
        neuron2 = QuantumNeuron("neuron2", complex(0, 1))
        
        neuron1.entangle_with(neuron2)
        
        assert "neuron2" in neuron1.entanglement_partners
        assert "neuron1" in neuron2.entanglement_partners


class TestQuantumNeuralNetwork:
    """Test quantum neural network functionality"""
    
    def test_quantum_network_initialization(self):
        """Test quantum neural network creation"""
        architecture = {
            "layer_sizes": [3, 5, 2],
            "quantum_gates": 10,
            "entanglement_density": 0.5
        }
        
        network = QuantumNeuralNetwork(architecture)
        
        assert len(network.neurons) == 10  # 3 + 5 + 2
        assert network.entanglement_matrix.shape == (10, 10)
        assert len(network.quantum_gates) == 0  # Initially empty
    
    def test_quantum_forward_pass(self):
        """Test quantum forward pass computation"""
        architecture = {"layer_sizes": [3, 4, 2]}
        network = QuantumNeuralNetwork(architecture)
        
        inputs = np.array([0.5, 0.3, 0.8])
        outputs = network.quantum_forward_pass(inputs)
        
        assert len(outputs) == 2  # Output layer size
        assert all(isinstance(output, (int, float)) for output in outputs)
        assert all(0 <= output <= 1 for output in outputs)  # Probability outputs


class TestNeuralGenome:
    """Test neural genome functionality"""
    
    def test_genome_creation(self):
        """Test neural genome creation"""
        architecture = {"num_layers": 4, "hidden_dim": 256}
        
        genome = NeuralGenome(
            genome_id="test_genome",
            network_type=NeuralNetworkType.TRANSFORMER,
            architecture=architecture
        )
        
        assert genome.genome_id == "test_genome"
        assert genome.network_type == NeuralNetworkType.TRANSFORMER
        assert genome.architecture == architecture
        assert genome.fitness == 0.0
        assert genome.generation == 0
        assert len(genome.parent_ids) == 0
    
    def test_fitness_calculation(self):
        """Test fitness calculation"""
        genome = NeuralGenome("test", NeuralNetworkType.LSTM, {})
        
        task_performance = {
            "accuracy": 0.8,
            "speed": 0.6,
            "efficiency": 0.7
        }
        
        fitness = genome.calculate_fitness(task_performance)
        
        assert 0 <= fitness <= 1
        assert len(genome.performance_history) == 1
        assert genome.performance_history[0] == fitness
    
    def test_diversity_bonus(self):
        """Test diversity bonus calculation"""
        genome = NeuralGenome("test", NeuralNetworkType.QUANTUM_NEURAL, {})
        
        # Add diverse mutations
        for i in range(15):
            genome.mutations.append({"type": f"mutation_{i % 3}"})
        
        bonus = genome._diversity_bonus()
        assert bonus > 0
        assert bonus <= 0.1


@pytest.mark.asyncio
class TestNeuralEvolutionEngine:
    """Test neural evolution engine functionality"""
    
    async def test_engine_initialization(self):
        """Test evolution engine initialization"""
        engine = NeuralEvolutionEngine(population_size=20)
        
        await engine.initialize_population()
        
        assert len(engine.population) == 20
        assert engine.generation == 0
        assert engine.population_size == 20
        assert len(engine.quantum_networks) > 0  # Should have some quantum networks
    
    async def test_population_diversity(self):
        """Test population has diverse network types"""
        engine = NeuralEvolutionEngine(population_size=30)
        await engine.initialize_population()
        
        network_types = [genome.network_type for genome in engine.population.values()]
        unique_types = set(network_types)
        
        assert len(unique_types) > 1  # Should have multiple network types
    
    async def test_fitness_evaluation(self):
        """Test fitness evaluation for population"""
        engine = NeuralEvolutionEngine(population_size=10)
        await engine.initialize_population()
        
        await engine._evaluate_population_fitness()
        
        # All genomes should have fitness scores
        for genome in engine.population.values():
            assert genome.fitness >= 0
            assert len(engine.task_performance_db[genome.genome_id]) > 0
    
    async def test_pattern_recognition_task(self):
        """Test pattern recognition task evaluation"""
        engine = NeuralEvolutionEngine()
        genome = NeuralGenome(
            "test", 
            NeuralNetworkType.CNN, 
            {"conv_layers": 3, "filters": [32, 64, 128]}
        )
        
        accuracy = await engine._test_pattern_recognition(genome)
        
        assert 0 <= accuracy <= 1
        # CNNs should get bonus for pattern recognition
        assert accuracy >= 0.6  # Should be reasonably high
    
    async def test_quantum_network_evaluation(self):
        """Test quantum neural network evaluation"""
        engine = NeuralEvolutionEngine()
        
        # Create quantum genome
        quantum_architecture = {
            "layer_sizes": [5, 8, 3],
            "quantum_gates": 12,
            "entanglement_density": 0.7
        }
        
        genome = NeuralGenome(
            "quantum_test",
            NeuralNetworkType.QUANTUM_NEURAL,
            quantum_architecture
        )
        
        # Create quantum network
        engine.quantum_networks[genome.genome_id] = QuantumNeuralNetwork(quantum_architecture)
        
        accuracy = await engine._test_pattern_recognition(genome)
        
        assert 0 <= accuracy <= 1
        # Should handle quantum network evaluation
    
    async def test_parent_selection(self):
        """Test parent selection for evolution"""
        engine = NeuralEvolutionEngine(population_size=20)
        await engine.initialize_population()
        
        # Set some fitness values
        for i, genome in enumerate(engine.population.values()):
            genome.fitness = i * 0.1  # Varying fitness
        
        parents = engine._select_parents()
        
        assert len(parents) >= 2  # Should select multiple parents
        # Higher fitness genomes should be more likely to be selected
        parent_fitnesses = [p.fitness for p in parents]
        assert max(parent_fitnesses) > 0.5  # Should include high-fitness parents
    
    async def test_crossover(self):
        """Test genetic crossover operation"""
        engine = NeuralEvolutionEngine()
        
        parent1 = NeuralGenome(
            "parent1",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 6, "hidden_dim": 512, "num_heads": 8}
        )
        
        parent2 = NeuralGenome(
            "parent2", 
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 4, "hidden_dim": 256, "num_heads": 4}
        )
        
        child1, child2 = await engine._crossover(parent1, parent2)
        
        assert child1.generation == engine.generation + 1
        assert child2.generation == engine.generation + 1
        assert parent1.genome_id in child1.parent_ids
        assert parent2.genome_id in child1.parent_ids
        assert len(child1.architecture) > 0
        assert len(child2.architecture) > 0
    
    async def test_mutation(self):
        """Test genetic mutation operation"""
        engine = NeuralEvolutionEngine()
        
        # Test transformer mutation
        transformer_genome = NeuralGenome(
            "transformer",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 6, "hidden_dim": 512}
        )
        
        original_arch = transformer_genome.architecture.copy()
        await engine._mutate_genome(transformer_genome)
        
        # Architecture should potentially be different
        # (mutation is probabilistic, so we just check it doesn't crash)
        assert isinstance(transformer_genome.architecture, dict)
        assert len(transformer_genome.mutations) <= 1  # Might or might not mutate
    
    async def test_quantum_mutation(self):
        """Test quantum-specific mutation"""
        engine = NeuralEvolutionEngine()
        
        quantum_genome = NeuralGenome(
            "quantum",
            NeuralNetworkType.QUANTUM_NEURAL,
            {"layer_sizes": [5, 8, 3], "quantum_gates": 10}
        )
        
        # Create quantum network
        engine.quantum_networks[quantum_genome.genome_id] = QuantumNeuralNetwork(quantum_genome.architecture)
        
        await engine._mutate_genome(quantum_genome)
        
        # Should handle quantum mutations without errors
        assert isinstance(quantum_genome.architecture, dict)
    
    async def test_population_replacement(self):
        """Test population replacement with new generation"""
        engine = NeuralEvolutionEngine(population_size=10)
        await engine.initialize_population()
        
        # Create offspring
        offspring = []
        for i in range(5):
            child = NeuralGenome(
                f"child_{i}",
                NeuralNetworkType.LSTM,
                {"hidden_size": 128, "num_layers": 2}
            )
            child.fitness = 0.8  # High fitness
            offspring.append(child)
        
        original_ids = set(engine.population.keys())
        engine._replace_population(offspring)
        
        assert len(engine.population) == engine.population_size
        # Some genomes should be new
        new_ids = set(engine.population.keys())
        assert len(new_ids & original_ids) < len(original_ids)  # Some replaced
    
    async def test_diversity_maintenance(self):
        """Test diversity maintenance mechanism"""
        engine = NeuralEvolutionEngine(population_size=20)
        await engine.initialize_population()
        
        # Force low diversity by making all genomes the same type
        for genome in engine.population.values():
            genome.network_type = NeuralNetworkType.LSTM
        
        diversity_score = engine._calculate_population_diversity()
        assert diversity_score < 0.5  # Should be low
        
        await engine._introduce_diverse_genomes()
        
        new_diversity = engine._calculate_population_diversity()
        assert new_diversity > diversity_score  # Should be improved
    
    async def test_quantum_coherence_maintenance(self):
        """Test quantum coherence maintenance"""
        engine = NeuralEvolutionEngine()
        
        # Create quantum network with low coherence
        architecture = {"layer_sizes": [3, 4, 2]}
        quantum_net = QuantumNeuralNetwork(architecture)
        
        # Set low coherence states
        for neuron in quantum_net.neurons.values():
            neuron.quantum_state = complex(0.05, 0.05)  # Very low coherence
        
        await engine._boost_network_coherence(quantum_net)
        
        # Coherence should be improved
        coherences = [abs(neuron.quantum_state) for neuron in quantum_net.neurons.values()]
        assert all(coherence > 0.1 for coherence in coherences)
    
    async def test_evolution_report(self):
        """Test evolution report generation"""
        engine = NeuralEvolutionEngine(population_size=15)
        await engine.initialize_population()
        await engine._evaluate_population_fitness()
        
        report = engine.get_evolution_report()
        
        assert "generation" in report
        assert "population_stats" in report
        assert "best_performer" in report
        assert "diversity_metrics" in report
        assert "evolution_trends" in report
        assert "quantum_coherence" in report
        
        # Check population stats
        pop_stats = report["population_stats"]
        assert pop_stats["size"] == 15
        assert "avg_fitness" in pop_stats
        assert "max_fitness" in pop_stats
        
        # Check diversity metrics
        diversity = report["diversity_metrics"]
        assert "overall_diversity" in diversity
        assert "network_type_distribution" in diversity
    
    async def test_novel_task_evaluation(self):
        """Test evaluation with novel tasks"""
        engine = NeuralEvolutionEngine()
        
        genome = NeuralGenome(
            "test",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 6, "hidden_dim": 512}
        )
        genome.fitness = 0.7
        
        novel_performance = await engine._evaluate_with_novel_tasks(genome)
        
        assert 0 <= novel_performance <= 1
        # Should be combination of existing and novel performance
    
    async def test_multimodal_capability(self):
        """Test multimodal capability evaluation"""
        engine = NeuralEvolutionEngine()
        
        # Transformer should perform well on multimodal tasks
        transformer_genome = NeuralGenome(
            "transformer",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 8}
        )
        
        multimodal_score = await engine._test_multimodal_capability(transformer_genome)
        assert 0 <= multimodal_score <= 1
        assert multimodal_score >= 0.6  # Transformers should score well
        
        # Quantum should perform even better
        quantum_genome = NeuralGenome(
            "quantum",
            NeuralNetworkType.QUANTUM_NEURAL,
            {"layer_sizes": [5, 8, 3]}
        )
        
        quantum_score = await engine._test_multimodal_capability(quantum_genome)
        assert quantum_score >= multimodal_score
    
    async def test_transfer_learning_evaluation(self):
        """Test transfer learning capability evaluation"""
        engine = NeuralEvolutionEngine()
        
        genome = NeuralGenome(
            "test",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 6}
        )
        
        # Add mutation history to test diversity bonus
        for i in range(8):
            genome.mutations.append({"type": f"mutation_{i}"})
        
        transfer_score = await engine._test_transfer_learning(genome)
        
        assert 0 <= transfer_score <= 1
        # Should get bonus for diverse mutations and transformer type
        assert transfer_score >= 0.6
    
    async def test_adversarial_robustness(self):
        """Test adversarial robustness evaluation"""
        engine = NeuralEvolutionEngine()
        
        # Test quantum network robustness
        quantum_genome = NeuralGenome(
            "quantum",
            NeuralNetworkType.QUANTUM_NEURAL,
            {"layer_sizes": [5, 5, 2]}
        )
        
        robustness = await engine._test_adversarial_robustness(quantum_genome)
        
        assert 0 <= robustness <= 1
        # Quantum networks should get robustness bonus
        assert robustness >= 0.6
    
    async def test_architecture_complexity(self):
        """Test architecture complexity calculation"""
        engine = NeuralEvolutionEngine()
        
        # Simple architecture
        simple_genome = NeuralGenome(
            "simple",
            NeuralNetworkType.LSTM,
            {"hidden_size": 64, "num_layers": 2}
        )
        
        simple_complexity = engine._calculate_architecture_complexity(simple_genome)
        
        # Complex architecture
        complex_genome = NeuralGenome(
            "complex",
            NeuralNetworkType.TRANSFORMER,
            {"num_layers": 24, "num_heads": 16, "hidden_dim": 2048}
        )
        
        complex_complexity = engine._calculate_architecture_complexity(complex_genome)
        
        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
        assert complex_complexity > simple_complexity
    
    async def test_rule_evolution(self):
        """Test genetic algorithm for rule evolution"""
        engine = NeuralEvolutionEngine(population_size=10)
        await engine.initialize_population()
        
        # Set up some genomes with different fitness levels
        genomes = list(engine.population.values())
        for i, genome in enumerate(genomes):
            genome.fitness = i * 0.1
        
        # Add to best performers to enable evolution
        engine.best_performers = genomes[:5]
        
        # Test rule crossover
        parent1 = genomes[0]
        parent2 = genomes[1]
        
        child_rule = engine._crossover_rules(
            type('MockRule', (), {
                'rule_id': 'parent1',
                'trigger_pattern': 'pattern1',
                'strategy': 'strategy1',
                'parameters': {'param1': 1.0, 'param2': 2.0},
                'success_rate': 0.8
            })(),
            type('MockRule', (), {
                'rule_id': 'parent2',
                'trigger_pattern': 'pattern2', 
                'strategy': 'strategy2',
                'parameters': {'param1': 1.5, 'param3': 3.0},
                'success_rate': 0.6
            })()
        )
        
        assert hasattr(child_rule, 'rule_id')
        assert hasattr(child_rule, 'parameters')
        assert len(child_rule.parameters) > 0


@pytest.mark.asyncio
class TestGlobalEvolutionEngine:
    """Test global evolution engine singleton"""
    
    async def test_get_evolution_engine(self):
        """Test global evolution engine creation"""
        # Reset global instance
        import pipeline.core.neural_evolution_engine as module
        module._evolution_engine = None
        
        engine1 = await get_evolution_engine()
        engine2 = await get_evolution_engine()
        
        assert engine1 is engine2  # Should be same instance
        assert len(engine1.population) > 0
    
    async def test_engine_lifecycle(self):
        """Test complete engine lifecycle"""
        engine = NeuralEvolutionEngine(population_size=5)
        
        # Initialize
        await engine.initialize_population()
        assert len(engine.population) == 5
        
        # Run one evolution cycle
        await engine._evolve_generation()
        assert engine.generation == 1
        
        # Shutdown
        await engine.shutdown()
        assert not engine._evolution_active


@pytest.mark.integration
class TestNeuralEvolutionIntegration:
    """Integration tests for neural evolution system"""
    
    async def test_full_evolution_cycle(self):
        """Test complete evolution cycle"""
        engine = NeuralEvolutionEngine(population_size=20)
        
        # Initialize and run for a few generations
        await engine.initialize_population()
        
        initial_fitness = [g.fitness for g in engine.population.values()]
        
        # Run evolution for 3 generations
        for _ in range(3):
            await engine._evolve_generation()
        
        final_fitness = [g.fitness for g in engine.population.values()]
        
        assert engine.generation == 3
        assert len(engine.evolution_history) == 3
        
        # Population should maintain size
        assert len(engine.population) == 20
        
        # Should have some quantum networks
        assert len(engine.quantum_networks) > 0
    
    async def test_performance_optimization(self):
        """Test that evolution improves performance over time"""
        engine = NeuralEvolutionEngine(population_size=15)
        await engine.initialize_population()
        
        # Track best fitness over generations
        best_fitness_history = []
        
        for generation in range(5):
            await engine._evolve_generation()
            
            current_best = max(g.fitness for g in engine.population.values())
            best_fitness_history.append(current_best)
        
        # Performance should generally improve or at least not degrade significantly
        assert best_fitness_history[-1] >= best_fitness_history[0] * 0.8
    
    async def test_diversity_preservation(self):
        """Test that evolution preserves genetic diversity"""
        engine = NeuralEvolutionEngine(population_size=25)
        await engine.initialize_population()
        
        initial_diversity = engine._calculate_population_diversity()
        
        # Run evolution
        for _ in range(4):
            await engine._evolve_generation()
        
        final_diversity = engine._calculate_population_diversity()
        
        # Should maintain reasonable diversity
        assert final_diversity >= 0.2
        # Shouldn't lose all diversity
        assert final_diversity >= initial_diversity * 0.5
    
    async def test_quantum_evolution_stability(self):
        """Test quantum neural networks remain stable during evolution"""
        engine = NeuralEvolutionEngine(population_size=10)
        await engine.initialize_population()
        
        # Count initial quantum networks
        initial_quantum_count = len(engine.quantum_networks)
        
        # Run evolution
        for _ in range(3):
            await engine._evolve_generation()
        
        # Should still have quantum networks
        assert len(engine.quantum_networks) > 0
        
        # Test quantum coherence
        for quantum_net in engine.quantum_networks.values():
            coherences = [abs(neuron.quantum_state) for neuron in quantum_net.neurons.values()]
            assert all(coherence >= 0 for coherence in coherences)
            assert any(coherence > 0.1 for coherence in coherences)  # Some coherent neurons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])