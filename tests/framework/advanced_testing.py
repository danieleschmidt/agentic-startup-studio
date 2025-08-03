"""
Advanced Testing Framework for Agentic Startup Studio.

This module provides sophisticated testing utilities including performance testing,
integration testing, property-based testing, and comprehensive mocking.
"""

import asyncio
import time
import statistics
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch
import pytest
import json
import tempfile
import os
from uuid import UUID, uuid4

from pipeline.models.idea import Idea, IdeaCategory, IdeaStatus, PipelineStage
from pipeline.services.idea_analytics_service import IdeaAnalyticsService
from pipeline.services.enhanced_evidence_collector import EnhancedEvidenceCollector
from pipeline.storage.base_repository import BaseRepository


@dataclass
class PerformanceMetrics:
    """Performance test results."""
    
    operation_name: str
    total_operations: int
    total_time_seconds: float
    average_time_seconds: float
    median_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    percentile_95_seconds: float
    percentile_99_seconds: float
    operations_per_second: float
    success_rate: float
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'total_operations': self.total_operations,
            'total_time_seconds': self.total_time_seconds,
            'average_time_seconds': self.average_time_seconds,
            'median_time_seconds': self.median_time_seconds,
            'min_time_seconds': self.min_time_seconds,
            'max_time_seconds': self.max_time_seconds,
            'percentile_95_seconds': self.percentile_95_seconds,
            'percentile_99_seconds': self.percentile_99_seconds,
            'operations_per_second': self.operations_per_second,
            'success_rate': self.success_rate,
            'error_count': self.error_count
        }


class PerformanceTester:
    """Performance testing utility with detailed metrics collection."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
    
    @asynccontextmanager
    async def measure_async_performance(
        self, 
        operation_name: str,
        target_operations: int = 100,
        max_duration_seconds: float = 60.0,
        concurrency: int = 1
    ) -> AsyncIterator[Callable]:
        """
        Measure performance of async operations.
        
        Args:
            operation_name: Name of the operation being tested
            target_operations: Target number of operations to execute
            max_duration_seconds: Maximum test duration
            concurrency: Number of concurrent operations
            
        Yields:
            Function to execute for performance measurement
        """
        execution_times = []
        error_count = 0
        start_time = time.time()
        
        async def execute_operation(operation_func: Callable) -> None:
            """Execute a single operation and record timing."""
            nonlocal error_count
            
            op_start = time.time()
            try:
                await operation_func()
            except Exception as e:
                error_count += 1
                # Don't raise to continue testing other operations
            finally:
                op_end = time.time()
                execution_times.append(op_end - op_start)
        
        # Yield the execution function
        yield execute_operation
        
        # Calculate metrics
        total_time = time.time() - start_time
        total_operations = len(execution_times)
        
        if total_operations > 0:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            # Calculate percentiles
            sorted_times = sorted(execution_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            
            p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_time
            p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_time
            
            ops_per_second = total_operations / total_time if total_time > 0 else 0
            success_rate = (total_operations - error_count) / total_operations * 100
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                total_operations=total_operations,
                total_time_seconds=total_time,
                average_time_seconds=avg_time,
                median_time_seconds=median_time,
                min_time_seconds=min_time,
                max_time_seconds=max_time,
                percentile_95_seconds=p95_time,
                percentile_99_seconds=p99_time,
                operations_per_second=ops_per_second,
                success_rate=success_rate,
                error_count=error_count
            )
            
            self.metrics_history.append(metrics)
    
    async def benchmark_concurrent_operations(
        self,
        operation_func: Callable,
        operation_name: str,
        concurrency_levels: List[int] = [1, 5, 10, 20],
        operations_per_level: int = 50
    ) -> Dict[int, PerformanceMetrics]:
        """
        Benchmark operation at different concurrency levels.
        
        Args:
            operation_func: Async function to benchmark
            operation_name: Name for the operation
            concurrency_levels: List of concurrency levels to test
            operations_per_level: Operations per concurrency level
            
        Returns:
            Dictionary mapping concurrency level to performance metrics
        """
        results = {}
        
        for concurrency in concurrency_levels:
            async with self.measure_async_performance(
                f"{operation_name}_concurrency_{concurrency}",
                target_operations=operations_per_level,
                concurrency=concurrency
            ) as execute_op:
                
                # Create concurrent tasks
                tasks = []
                for _ in range(operations_per_level):
                    task = asyncio.create_task(execute_op(operation_func))
                    tasks.append(task)
                
                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Get the latest metrics
            if self.metrics_history:
                results[concurrency] = self.metrics_history[-1]
        
        return results
    
    def save_metrics_report(self, filepath: str) -> None:
        """Save performance metrics to JSON file."""
        metrics_data = [metrics.to_dict() for metrics in self.metrics_history]
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics_data
            }, f, indent=2)


class PropertyBasedTester:
    """Property-based testing utilities using Hypothesis-style testing."""
    
    @staticmethod
    def generate_valid_idea(
        category: Optional[IdeaCategory] = None,
        status: Optional[IdeaStatus] = None,
        stage: Optional[PipelineStage] = None
    ) -> Idea:
        """Generate a valid Idea instance with optional constraints."""
        import random
        
        categories = list(IdeaCategory) if category is None else [category]
        statuses = list(IdeaStatus) if status is None else [status]
        stages = list(PipelineStage) if stage is None else [stage]
        
        return Idea(
            title=f"Test Idea {random.randint(1000, 9999)}",
            description=f"This is a test description for a {random.choice(categories).value} idea. " * 3,
            category=random.choice(categories),
            status=random.choice(statuses),
            current_stage=random.choice(stages),
            problem_statement=f"Problem statement for test idea {random.randint(1, 100)}",
            solution_description=f"Solution description for test idea {random.randint(1, 100)}",
            target_market=f"Target market {random.randint(1, 10)}"
        )
    
    @staticmethod
    def generate_idea_variations(base_idea: Idea, count: int = 10) -> List[Idea]:
        """Generate variations of a base idea for property testing."""
        import random
        
        variations = []
        
        for i in range(count):
            # Create a copy with variations
            idea = Idea(
                title=f"{base_idea.title} - Variation {i+1}",
                description=f"{base_idea.description} Additional context {i+1}.",
                category=base_idea.category,
                status=base_idea.status,
                current_stage=base_idea.current_stage,
                problem_statement=base_idea.problem_statement,
                solution_description=f"{base_idea.solution_description} Variation {i+1}.",
                target_market=base_idea.target_market
            )
            variations.append(idea)
        
        return variations
    
    @staticmethod
    def test_invariant(func: Callable[[Any], bool], test_data: List[Any]) -> List[Any]:
        """
        Test an invariant function against test data.
        
        Args:
            func: Function that should return True for all valid inputs
            test_data: List of test inputs
            
        Returns:
            List of inputs that failed the invariant
        """
        failures = []
        
        for data in test_data:
            try:
                if not func(data):
                    failures.append(data)
            except Exception as e:
                failures.append((data, str(e)))
        
        return failures


class IntegrationTestHarness:
    """Comprehensive integration testing harness."""
    
    def __init__(self):
        self.mocked_services = {}
        self.test_data_cleanup = []
    
    @asynccontextmanager
    async def mock_external_services(self):
        """Mock external service dependencies."""
        mocks = {
            'openai': AsyncMock(),
            'google_ai': AsyncMock(),
            'search_api': AsyncMock(),
            'database': AsyncMock()
        }
        
        # Configure mock responses
        mocks['openai'].complete.return_value = "Mock OpenAI response"
        mocks['google_ai'].generate.return_value = "Mock Google AI response"
        mocks['search_api'].search.return_value = [
            {'url': 'https://example.com', 'title': 'Mock Result', 'snippet': 'Mock snippet'}
        ]
        
        with patch.multiple(
            'pipeline.services',
            openai_client=mocks['openai'],
            google_ai_client=mocks['google_ai'],
            search_client=mocks['search_api']
        ):
            self.mocked_services = mocks
            yield mocks
    
    @asynccontextmanager
    async def isolated_test_environment(self):
        """Create isolated test environment with cleanup."""
        # Create temporary directory for test files
        temp_dir = tempfile.mkdtemp(prefix='test_agentic_')
        
        try:
            # Set test environment variables
            test_env = {
                'ENVIRONMENT': 'test',
                'DATABASE_URL': 'sqlite:///:memory:',
                'LOG_LEVEL': 'DEBUG',
                'TEMP_DIR': temp_dir
            }
            
            with patch.dict(os.environ, test_env):
                yield temp_dir
        finally:
            # Cleanup test data
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def setup_test_data(self) -> Dict[str, Any]:
        """Setup comprehensive test data."""
        test_ideas = [
            PropertyBasedTester.generate_valid_idea(
                category=IdeaCategory.AI_ML,
                status=IdeaStatus.RESEARCHING,
                stage=PipelineStage.RESEARCH
            )
            for _ in range(5)
        ]
        
        test_data = {
            'ideas': test_ideas,
            'users': ['test_user_1', 'test_user_2', 'admin_user'],
            'timestamps': {
                'start': datetime.now(timezone.utc),
                'data_created': datetime.now(timezone.utc)
            }
        }
        
        self.test_data_cleanup.append(test_data)
        return test_data
    
    async def cleanup_test_data(self):
        """Cleanup all test data."""
        for test_data in self.test_data_cleanup:
            # Cleanup would happen here in a real implementation
            pass
        self.test_data_cleanup.clear()


class APITestClient:
    """Advanced API testing client with comprehensive features."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_headers = {}
        self.request_history = []
    
    def set_auth_token(self, token: str):
        """Set authentication token for requests."""
        self.session_headers['Authorization'] = f"Bearer {token}"
    
    async def request(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        headers: Dict[str, str] = None,
        timeout: float = 30.0,
        expect_status: int = 200
    ) -> Dict[str, Any]:
        """Make HTTP request with comprehensive logging and validation."""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}"
        request_headers = {**self.session_headers}
        if headers:
            request_headers.update(headers)
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=data if data else None,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    response_time = time.time() - start_time
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    # Log request details
                    request_log = {
                        'method': method,
                        'url': url,
                        'status_code': response.status,
                        'response_time': response_time,
                        'request_data': data,
                        'response_data': response_data,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    self.request_history.append(request_log)
                    
                    # Validate expected status
                    if response.status != expect_status:
                        raise AssertionError(
                            f"Expected status {expect_status}, got {response.status}. "
                            f"Response: {response_data}"
                        )
                    
                    return {
                        'status_code': response.status,
                        'data': response_data,
                        'headers': dict(response.headers),
                        'response_time': response_time
                    }
        
        except Exception as e:
            error_log = {
                'method': method,
                'url': url,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.request_history.append(error_log)
            raise
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request('GET', endpoint, **kwargs)
    
    async def post(self, endpoint: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request('POST', endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request('PUT', endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request('DELETE', endpoint, **kwargs)
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get history of all requests made."""
        return self.request_history.copy()
    
    def clear_history(self):
        """Clear request history."""
        self.request_history.clear()


class TestDataFactory:
    """Factory for generating comprehensive test data."""
    
    @staticmethod
    def create_research_data() -> Dict[str, Any]:
        """Create mock research data."""
        return {
            "market_size": {"value": 2500.0, "currency": "USD", "unit": "millions"},
            "growth_rate": 0.18,
            "trends": ["AI automation", "Remote work", "Digital transformation"],
            "competitors": {
                "direct": [
                    {"name": "Competitor A", "market_share": 0.15, "funding": "50M"},
                    {"name": "Competitor B", "market_share": 0.08, "funding": "25M"}
                ],
                "indirect": [
                    {"name": "Big Tech Co", "market_share": 0.30, "category": "platform"}
                ]
            },
            "regulatory": {
                "complexity": "medium",
                "key_regulations": ["GDPR", "SOX"],
                "compliance_cost": "moderate"
            },
            "technology": {
                "readiness_level": 8,
                "key_technologies": ["Machine Learning", "Cloud Computing", "APIs"],
                "implementation_complexity": "medium"
            },
            "sources": [
                "Industry Report 2025",
                "Market Research Firm",
                "Academic Study",
                "Government Data"
            ]
        }
    
    @staticmethod
    def create_analytics_results() -> Dict[str, Any]:
        """Create mock analytics results."""
        return {
            "market_potential": {
                "overall_score": 0.82,
                "market_size_score": 0.85,
                "competition_score": 0.78,
                "timing_score": 0.80,
                "feasibility_score": 0.85,
                "innovation_score": 0.88,
                "confidence_level": 0.75
            },
            "competitive_analysis": {
                "market_gap_score": 0.72,
                "differentiation_score": 0.80,
                "competitive_advantages": [
                    "AI-powered automation",
                    "First-mover advantage",
                    "Strong technical team"
                ],
                "competitive_risks": [
                    "Large incumbents",
                    "Technology commoditization"
                ]
            },
            "funding_potential": {
                "overall_funding_score": 0.78,
                "estimated_funding_range": [500000, 2000000],
                "recommended_funding_stage": "seed",
                "key_metrics_needed": [
                    "User traction",
                    "Revenue model validation",
                    "Product-market fit"
                ]
            }
        }
    
    @staticmethod
    def create_workflow_execution() -> Dict[str, Any]:
        """Create mock workflow execution data."""
        return {
            "workflow_type": "idea_validation",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
            "steps_completed": [
                "validation",
                "research", 
                "analysis",
                "deck_generation"
            ],
            "performance_metrics": {
                "total_time_seconds": 900,
                "memory_usage_mb": 256,
                "api_calls_made": 15
            }
        }


# Pytest fixtures using the advanced testing framework

@pytest.fixture
async def performance_tester():
    """Provide performance testing utilities."""
    return PerformanceTester()


@pytest.fixture
async def property_tester():
    """Provide property-based testing utilities."""
    return PropertyBasedTester()


@pytest.fixture
async def integration_harness():
    """Provide integration testing harness."""
    harness = IntegrationTestHarness()
    yield harness
    await harness.cleanup_test_data()


@pytest.fixture
async def api_client():
    """Provide API testing client."""
    return APITestClient()


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory()


@pytest.fixture
async def mock_analytics_service():
    """Provide mocked analytics service."""
    service = AsyncMock(spec=IdeaAnalyticsService)
    
    # Configure default responses
    from pipeline.services.idea_analytics_service import MarketPotentialScore, CompetitiveAnalysis, FundingPotentialScore
    
    service.analyze_market_potential.return_value = MarketPotentialScore(
        overall_score=0.8,
        market_size_score=0.85,
        competition_score=0.75,
        timing_score=0.80,
        feasibility_score=0.85,
        innovation_score=0.88,
        confidence_level=0.75
    )
    
    service.analyze_competitive_landscape.return_value = CompetitiveAnalysis(
        market_gap_score=0.72,
        differentiation_score=0.80
    )
    
    service.calculate_funding_potential.return_value = FundingPotentialScore(
        overall_funding_score=0.78,
        stage_alignment_score=0.80,
        investor_appeal_score=0.75,
        scalability_score=0.80,
        team_readiness_score=0.70,
        estimated_funding_range=(500000, 2000000),
        recommended_funding_stage="seed"
    )
    
    return service


@pytest.fixture
async def mock_evidence_collector():
    """Provide mocked evidence collector."""
    collector = AsyncMock(spec=EnhancedEvidenceCollector)
    
    # Configure default response
    from pipeline.services.enhanced_evidence_collector import ComprehensiveEvidence, MarketEvidence, TechnicalEvidence, BusinessEvidence
    
    collector.collect_comprehensive_evidence.return_value = ComprehensiveEvidence(
        idea_id=uuid4(),
        market_evidence=MarketEvidence(),
        technical_evidence=TechnicalEvidence(),
        business_evidence=BusinessEvidence(),
        overall_confidence=0.75,
        evidence_quality_score=0.80,
        collection_timestamp=datetime.now(timezone.utc),
        summary="Comprehensive evidence collected successfully",
        key_insights=["Strong market opportunity", "Technical feasibility confirmed"],
        risk_factors=["Competitive pressure", "Regulatory uncertainty"],
        opportunities=["Market gap identified", "Technology advancement"]
    )
    
    return collector