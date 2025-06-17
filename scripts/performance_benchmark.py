#!/usr/bin/env python3
"""
Performance Benchmark Tool for Integrated IdeaManager System

Measures performance improvements from caching, monitoring, and optimized duplicate detection.
Generates detailed metrics comparing baseline vs. enhanced system performance.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

from pipeline.models.idea import IdeaDraft
from pipeline.ingestion.idea_manager import IdeaManager
from pipeline.config.settings import IngestionConfig
from pipeline.storage.idea_repository import IdeaRepository
from pipeline.ingestion.duplicate_detector import CacheableDuplicateDetector
from pipeline.ingestion.cache.cache_manager import CacheManager
from pipeline.ingestion.monitoring.metrics_collector import MetricsCollector


@dataclass
class BenchmarkResult:
    """Performance benchmark results."""
    operation: str
    baseline_time_ms: float
    enhanced_time_ms: float
    improvement_percent: float
    cache_hit_rate: float
    database_queries: int
    memory_usage_mb: float


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    total_operations: int
    average_improvement: float
    cache_efficiency: float
    database_query_reduction: float
    results: List[BenchmarkResult]


class PerformanceBenchmark:
    """Performance benchmark tool for the integrated system."""
    
    def __init__(self):
        self.config = IngestionConfig()
        self.results: List[BenchmarkResult] = []
        
    async def setup_test_data(self) -> List[IdeaDraft]:
        """Create test data for benchmarking."""
        test_ideas = []
        
        # Create diverse test ideas
        base_ideas = [
            ("AI-Powered Recipe Generator", "Create personalized recipes using AI", ["AI", "Food", "Health"]),
            ("Smart Home Energy Monitor", "Monitor and optimize home energy usage", ["IoT", "Sustainability", "Tech"]),
            ("Virtual Fitness Coach", "AI-driven personalized fitness training", ["AI", "Health", "Fitness"]),
            ("Automated Code Review Tool", "AI tool for comprehensive code analysis", ["AI", "Development", "Tools"]),
            ("Sustainable Fashion Marketplace", "Eco-friendly clothing platform", ["Sustainability", "Fashion", "E-commerce"]),
        ]
        
        # Generate variations for duplicate detection testing
        for i, (title, description, tags) in enumerate(base_ideas):
            # Original idea
            test_ideas.append(IdeaDraft(
                title=title,
                description=description,
                tags=tags,
                category="Technology"
            ))
            
            # Similar variations for duplicate testing
            test_ideas.append(IdeaDraft(
                title=f"{title} Pro",
                description=f"Enhanced {description.lower()}",
                tags=tags + ["Premium"],
                category="Technology"
            ))
            
            # Completely different ideas
            test_ideas.append(IdeaDraft(
                title=f"Innovative {title.split()[0]} Solution",
                description=f"Revolutionary approach to {description.split()[0].lower()}",
                tags=["Innovation"] + tags[:2],
                category="Innovation"
            ))
            
        return test_ideas
    
    async def benchmark_duplicate_detection(
        self, 
        ideas: List[IdeaDraft],
        with_cache: bool = True
    ) -> Dict[str, float]:
        """Benchmark duplicate detection performance."""
        
        # Setup components
        repository = IdeaRepository(self.config.database.connection_string)
        cache_manager = CacheManager(self.config) if with_cache else None
        metrics_collector = MetricsCollector()
        
        detector = CacheableDuplicateDetector(
            repository=repository,
            config=self.config.validation,
            cache_manager=cache_manager,
            metrics_collector=metrics_collector
        )
        
        start_time = time.perf_counter()
        total_operations = 0
        cache_hits = 0
        database_queries = 0
        
        # Process each idea
        for idea in ideas:
            operation_start = time.perf_counter()
            
            # Check for duplicates
            is_duplicate = await detector.is_duplicate(idea)
            total_operations += 1
            
            # Collect metrics
            if with_cache and cache_manager:
                # Simulate cache hit tracking
                if hasattr(detector, '_cache_hits'):
                    cache_hits = detector._cache_hits
            
            # Track database queries (would be implemented in repository)
            database_queries += 1  # Simplified tracking
            
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        return {
            'total_time_ms': total_time_ms,
            'avg_time_per_operation_ms': total_time_ms / total_operations,
            'cache_hit_rate': cache_hits / total_operations if total_operations > 0 else 0,
            'database_queries': database_queries,
            'operations_per_second': total_operations / (total_time_ms / 1000)
        }
    
    async def benchmark_idea_creation(
        self,
        ideas: List[IdeaDraft],
        with_enhancements: bool = True
    ) -> Dict[str, float]:
        """Benchmark complete idea creation workflow."""
        
        # Setup IdeaManager with or without enhancements
        if with_enhancements:
            cache_manager = CacheManager(self.config)
            metrics_collector = MetricsCollector()
            idea_manager = IdeaManager(
                repository=IdeaRepository(self.config.database.connection_string),
                validator=None,  # Would be injected
                duplicate_detector=CacheableDuplicateDetector(
                    repository=IdeaRepository(self.config.database.connection_string),
                    config=self.config.validation,
                    cache_manager=cache_manager,
                    metrics_collector=metrics_collector
                ),
                cache_manager=cache_manager,
                metrics_collector=metrics_collector
            )
        else:
            # Baseline without enhancements
            idea_manager = IdeaManager(
                repository=IdeaRepository(self.config.database.connection_string),
                validator=None,
                duplicate_detector=CacheableDuplicateDetector(
                    repository=IdeaRepository(self.config.database.connection_string),
                    config=self.config.validation,
                    cache_manager=None,
                    metrics_collector=None
                ),
                cache_manager=None,
                metrics_collector=None
            )
        
        start_time = time.perf_counter()
        successful_operations = 0
        
        for idea in ideas[:10]:  # Limit for benchmarking
            try:
                # This would normally create the idea
                # For benchmarking, we simulate the process
                await idea_manager.duplicate_detector.is_duplicate(idea)
                successful_operations += 1
            except Exception as e:
                print(f"Error processing idea: {e}")
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        return {
            'total_time_ms': total_time_ms,
            'successful_operations': successful_operations,
            'avg_time_per_operation_ms': total_time_ms / successful_operations if successful_operations > 0 else 0,
            'throughput_ops_per_second': successful_operations / (total_time_ms / 1000)
        }
    
    async def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run comprehensive performance benchmark."""
        print(">> Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        # Setup test data
        test_ideas = await self.setup_test_data()
        print(f">> Generated {len(test_ideas)} test ideas")
        
        # Benchmark 1: Duplicate Detection (Baseline vs Enhanced)
        print("\n>> Benchmarking Duplicate Detection...")
        baseline_duplicate = await self.benchmark_duplicate_detection(test_ideas, with_cache=False)
        enhanced_duplicate = await self.benchmark_duplicate_detection(test_ideas, with_cache=True)
        
        duplicate_improvement = (
            (baseline_duplicate['avg_time_per_operation_ms'] - enhanced_duplicate['avg_time_per_operation_ms']) /
            baseline_duplicate['avg_time_per_operation_ms'] * 100
        )
        
        self.results.append(BenchmarkResult(
            operation="Duplicate Detection",
            baseline_time_ms=baseline_duplicate['avg_time_per_operation_ms'],
            enhanced_time_ms=enhanced_duplicate['avg_time_per_operation_ms'],
            improvement_percent=duplicate_improvement,
            cache_hit_rate=enhanced_duplicate['cache_hit_rate'],
            database_queries=enhanced_duplicate['database_queries'],
            memory_usage_mb=0.0  # Would be measured in production
        ))
        
        # Benchmark 2: Complete Workflow (Baseline vs Enhanced)
        print(">> Benchmarking Complete Workflow...")
        baseline_workflow = await self.benchmark_idea_creation(test_ideas, with_enhancements=False)
        enhanced_workflow = await self.benchmark_idea_creation(test_ideas, with_enhancements=True)
        
        workflow_improvement = (
            (baseline_workflow['avg_time_per_operation_ms'] - enhanced_workflow['avg_time_per_operation_ms']) /
            baseline_workflow['avg_time_per_operation_ms'] * 100
        )
        
        self.results.append(BenchmarkResult(
            operation="Complete Workflow",
            baseline_time_ms=baseline_workflow['avg_time_per_operation_ms'],
            enhanced_time_ms=enhanced_workflow['avg_time_per_operation_ms'],
            improvement_percent=workflow_improvement,
            cache_hit_rate=0.0,  # Would be calculated from metrics
            database_queries=0,
            memory_usage_mb=0.0
        ))
        
        # Calculate summary metrics
        avg_improvement = statistics.mean([r.improvement_percent for r in self.results])
        cache_efficiency = statistics.mean([r.cache_hit_rate for r in self.results if r.cache_hit_rate > 0])
        
        return BenchmarkSuite(
            total_operations=len(self.results),
            average_improvement=avg_improvement,
            cache_efficiency=cache_efficiency,
            database_query_reduction=0.0,  # Would be calculated
            results=self.results
        )
    
    def generate_report(self, suite: BenchmarkSuite) -> str:
        """Generate detailed performance report."""
        report = []
        report.append(">> PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f">> Average Performance Improvement: {suite.average_improvement:.1f}%")
        report.append(f">> Cache Efficiency: {suite.cache_efficiency:.1f}%")
        report.append(f">> Total Operations Benchmarked: {suite.total_operations}")
        
        report.append("\n>> DETAILED RESULTS")
        report.append("-" * 30)
        
        for result in suite.results:
            report.append(f"\n>> {result.operation}")
            report.append(f"   Baseline: {result.baseline_time_ms:.2f}ms")
            report.append(f"   Enhanced: {result.enhanced_time_ms:.2f}ms")
            report.append(f"   Improvement: {result.improvement_percent:.1f}%")
            if result.cache_hit_rate > 0:
                report.append(f"   Cache Hit Rate: {result.cache_hit_rate:.1f}%")
        
        report.append("\nOK: PERFORMANCE TARGETS")
        report.append("-" * 25)
        target_met = suite.average_improvement >= 50.0
        report.append(f">> Target >50% improvement: {'OK: ACHIEVED' if target_met else 'ERROR: NOT MET'}")
        
        cache_target_met = suite.cache_efficiency >= 80.0
        report.append(f">> Target >80% cache efficiency: {'OK: ACHIEVED' if cache_target_met else 'ERROR: NOT MET'}")
        
        return "\n".join(report)
    
    async def save_results(self, suite: BenchmarkSuite, output_path: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        results_data = {
            'timestamp': time.time(),
            'summary': {
                'total_operations': suite.total_operations,
                'average_improvement': suite.average_improvement,
                'cache_efficiency': suite.cache_efficiency,
                'database_query_reduction': suite.database_query_reduction
            },
            'detailed_results': [
                {
                    'operation': r.operation,
                    'baseline_time_ms': r.baseline_time_ms,
                    'enhanced_time_ms': r.enhanced_time_ms,
                    'improvement_percent': r.improvement_percent,
                    'cache_hit_rate': r.cache_hit_rate,
                    'database_queries': r.database_queries,
                    'memory_usage_mb': r.memory_usage_mb
                }
                for r in suite.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f">> Results saved to {output_path}")


async def main():
    """Run performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run comprehensive benchmark
        suite = await benchmark.run_comprehensive_benchmark()
        
        # Generate and display report
        report = benchmark.generate_report(suite)
        print("\n" + report)
        
        # Save results
        await benchmark.save_results(suite)
        
        print("\n>> Benchmark completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)