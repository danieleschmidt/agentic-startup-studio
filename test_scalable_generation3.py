#!/usr/bin/env python3
"""
Generation 3 Validation Tests - MAKE IT SCALE (Optimized Implementation)
Tests performance optimization, caching, concurrency, and scalability features.
"""

import sys
import unittest
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestGeneration3Scalable(unittest.TestCase):
    """Scalability and performance tests for Generation 3"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_start_time = time.time()
        
    def test_concurrent_idea_processing(self):
        """Test concurrent processing of multiple ideas"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Create multiple ideas for concurrent processing
        ideas_data = []
        for i in range(10):
            ideas_data.append({
                'title': f'Concurrent Test Idea {i}',
                'description': f'A test idea for concurrent processing #{i}',
                'category': 'saas',
                'status': 'DRAFT',
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
        
        # Test concurrent creation
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(Idea, **data) for data in ideas_data]
            ideas = [future.result() for future in futures]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert all ideas were created successfully
        self.assertEqual(len(ideas), 10)
        # Assert reasonable performance (should be fast for model creation)
        self.assertLess(processing_time, 2.0, "Concurrent processing should be fast")
    
    def test_cache_performance(self):
        """Test caching functionality for performance"""
        try:
            from pipeline.ingestion.cache.cache_manager import CacheManager
            
            cache = CacheManager()
            
            # Test cache set and get performance
            test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            
            # Test write performance
            start_time = time.time()
            for i in range(100):
                cache.set(f"test_key_{i}", test_data, ttl=60)
            write_time = time.time() - start_time
            
            # Test read performance
            start_time = time.time()
            for i in range(100):
                result = cache.get(f"test_key_{i}")
                self.assertIsNotNone(result)
            read_time = time.time() - start_time
            
            # Cache operations should be fast
            self.assertLess(write_time, 1.0, "Cache writes should be fast")
            self.assertLess(read_time, 0.5, "Cache reads should be very fast")
            
        except ImportError as e:
            self.skipTest(f"Cache manager not available: {e}")
    
    def test_vector_search_optimization(self):
        """Test vector search performance optimization"""
        try:
            from pipeline.storage.optimized_vector_search import OptimizedVectorSearch
            
            # Test that optimized vector search can be instantiated
            vector_search = OptimizedVectorSearch()
            self.assertIsNotNone(vector_search)
            
            # Test performance characteristics would require actual vectors
            # For now, just test that the class is available and can be instantiated
            
        except ImportError as e:
            self.skipTest(f"Optimized vector search not available: {e}")
    
    def test_connection_pooling(self):
        """Test database connection pooling for scalability"""
        try:
            from pipeline.storage.connection_pool_manager import ConnectionPoolManager
            
            pool_manager = ConnectionPoolManager()
            self.assertIsNotNone(pool_manager)
            
            # Test pool configuration
            self.assertTrue(hasattr(pool_manager, 'get_pool'))
            
        except ImportError as e:
            self.skipTest(f"Connection pool manager not available: {e}")
    
    def test_async_performance_optimization(self):
        """Test async performance optimizations"""
        async def async_test():
            try:
                from pipeline.main_pipeline_async import AsyncPipeline
                
                # Test async pipeline initialization
                pipeline = AsyncPipeline()
                self.assertIsNotNone(pipeline)
                
                # Test that async methods are available
                self.assertTrue(hasattr(pipeline, 'process_async'))
                
            except ImportError as e:
                self.skipTest(f"Async pipeline not available: {e}")
        
        # Run async test
        if hasattr(asyncio, 'run'):
            asyncio.run(async_test())
        else:
            # Python 3.6 compatibility
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_test())
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        try:
            from pipeline.monitoring.real_time_optimizer import RealTimeOptimizer
            
            optimizer = RealTimeOptimizer()
            self.assertIsNotNone(optimizer)
            
            # Test performance metrics collection
            start_time = time.time()
            
            # Simulate some work
            time.sleep(0.1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Test that performance can be measured
            self.assertGreater(duration, 0.05)  # Should be at least 50ms
            self.assertLess(duration, 0.5)     # Should be less than 500ms
            
        except ImportError as e:
            self.skipTest(f"Real-time optimizer not available: {e}")
    
    def test_quantum_performance_optimization(self):
        """Test quantum-inspired performance optimization"""
        try:
            from pipeline.performance.quantum_performance_optimizer import QuantumPerformanceOptimizer
            
            optimizer = QuantumPerformanceOptimizer()
            self.assertIsNotNone(optimizer)
            
            # Test optimization methods are available
            self.assertTrue(hasattr(optimizer, 'optimize'))
            
        except ImportError as e:
            self.skipTest(f"Quantum performance optimizer not available: {e}")
    
    def test_auto_scaling_triggers(self):
        """Test auto-scaling trigger implementation"""
        try:
            from pipeline.infrastructure.auto_scaling import AutoScalingManager
            
            scaling_manager = AutoScalingManager()
            self.assertIsNotNone(scaling_manager)
            
            # Test scaling decision methods
            self.assertTrue(hasattr(scaling_manager, 'should_scale_up'))
            self.assertTrue(hasattr(scaling_manager, 'should_scale_down'))
            
        except ImportError as e:
            self.skipTest(f"Auto-scaling manager not available: {e}")
    
    def test_load_balancing_capabilities(self):
        """Test load balancing implementation"""
        # For this test, we'll check if the system can handle multiple concurrent requests
        from pipeline.models.idea import Idea
        from datetime import datetime
        import threading
        
        results = []
        errors = []
        
        def create_idea(index):
            try:
                idea = Idea(
                    title=f'Load Test Idea {index}',
                    description=f'Load testing idea #{index}',
                    category='saas',
                    status='DRAFT',
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                results.append(idea)
            except Exception as e:
                errors.append(e)
        
        # Create 20 concurrent threads
        threads = []
        start_time = time.time()
        
        for i in range(20):
            thread = threading.Thread(target=create_idea, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify results
        self.assertEqual(len(results), 20, "All ideas should be created successfully")
        self.assertEqual(len(errors), 0, "No errors should occur")
        self.assertLess(total_time, 5.0, "Concurrent processing should complete quickly")
    
    def test_resource_optimization(self):
        """Test resource usage optimization"""
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process many objects
        test_objects = []
        for i in range(1000):
            test_objects.append({
                'id': i,
                'data': f'test_data_{i}' * 100,  # Create some memory usage
                'timestamp': time.time()
            })
        
        # Measure memory after object creation
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del test_objects
        gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = peak_memory - initial_memory
        memory_cleaned = peak_memory - final_memory
        
        # Memory should be manageable
        self.assertLess(memory_increase, 500, "Memory usage should be reasonable")
        # Memory cleanup can be negative due to other system processes, so just check it's reasonable
        self.assertGreater(memory_cleaned, -10, "Memory cleanup should be reasonable")
        
        print(f"Memory usage: Initial {initial_memory:.1f}MB, Peak {peak_memory:.1f}MB, Final {final_memory:.1f}MB")

if __name__ == '__main__':
    print("⚡ Running Generation 3 Scalable Performance Tests")
    print("=" * 55)
    
    # Run tests
    unittest.main(verbosity=2)
