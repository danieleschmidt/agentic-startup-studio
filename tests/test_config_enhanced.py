"""Enhanced test configuration with comprehensive test settings and utilities."""

import os
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Test environment configuration
TEST_CONFIG = {
    "environment": "testing",
    "database": {
        "url": "postgresql://test:test@localhost:5432/test_agentic_studio",
        "pool_size": 5,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "echo": False
    },
    "redis": {
        "url": "redis://localhost:6379/1",
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5
    },
    "ai_services": {
        "openai": {
            "api_key": "test-key-openai",
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.1,
            "request_timeout": 30
        },
        "google_ai": {
            "api_key": "test-key-google",
            "model": "gemini-pro",
            "max_tokens": 1000,
            "temperature": 0.1
        }
    },
    "pipeline": {
        "batch_size": 5,
        "max_retries": 2,
        "timeout_seconds": 30,
        "quality_threshold": 0.7
    },
    "testing": {
        "timeout_seconds": 30,
        "max_test_duration": 300,
        "performance_threshold_ms": 1000,
        "memory_limit_mb": 500,
        "parallel_workers": 4
    },
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "disable_existing_loggers": False
    }
}


class TestEnvironmentManager:
    """Manage test environment setup and teardown."""
    
    def __init__(self):
        self.original_env = {}
        self.mock_patches = []
    
    def setup_test_environment(self):
        """Set up isolated test environment."""
        # Store original environment variables
        test_env_vars = {
            "ENVIRONMENT": "testing",
            "DATABASE_URL": TEST_CONFIG["database"]["url"],
            "REDIS_URL": TEST_CONFIG["redis"]["url"],
            "OPENAI_API_KEY": TEST_CONFIG["ai_services"]["openai"]["api_key"],
            "GOOGLE_AI_API_KEY": TEST_CONFIG["ai_services"]["google_ai"]["api_key"],
            "LOG_LEVEL": "DEBUG"
        }
        
        for key, value in test_env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
    
    def teardown_test_environment(self):
        """Restore original environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Stop all mock patches
        for patch_obj in self.mock_patches:
            patch_obj.stop()
        self.mock_patches.clear()
    
    def add_mock_patch(self, target: str, mock_obj: Any = None):
        """Add a mock patch that will be cleaned up automatically."""
        mock_obj = mock_obj or MagicMock()
        patch_obj = patch(target, mock_obj)
        patch_obj.start()
        self.mock_patches.append(patch_obj)
        return mock_obj


class TestDataManager:
    """Manage test data lifecycle."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> Path:
        """Create temporary file for testing."""
        import tempfile
        
        fd, path = tempfile.mkstemp(suffix=suffix)
        path_obj = Path(path)
        
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        self.temp_files.append(path_obj)
        return path_obj
    
    def create_temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        import tempfile
        
        path = Path(tempfile.mkdtemp())
        self.temp_dirs.append(path)
        return path
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        import shutil
        
        for file_path in self.temp_files:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        
        for dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
            except Exception:
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()


class TestMetricsCollector:
    """Collect and analyze test metrics."""
    
    def __init__(self):
        self.metrics = {
            "test_runs": 0,
            "test_failures": 0,
            "test_duration_total": 0.0,
            "test_duration_avg": 0.0,
            "memory_usage_peak": 0.0,
            "performance_violations": 0,
            "tests_by_category": {},
            "slowest_tests": []
        }
    
    def record_test_start(self, test_name: str, category: str = "unit"):
        """Record test start metrics."""
        import time
        import psutil
        import os
        
        self.metrics["test_runs"] += 1
        self.metrics["tests_by_category"][category] = (
            self.metrics["tests_by_category"].get(category, 0) + 1
        )
        
        # Record memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "start_time": time.time(),
            "start_memory_mb": memory_mb,
            "test_name": test_name,
            "category": category
        }
    
    def record_test_end(self, test_info: Dict[str, Any], success: bool = True):
        """Record test completion metrics."""
        import time
        import psutil
        import os
        
        end_time = time.time()
        duration = end_time - test_info["start_time"]
        
        # Update duration metrics
        self.metrics["test_duration_total"] += duration
        self.metrics["test_duration_avg"] = (
            self.metrics["test_duration_total"] / self.metrics["test_runs"]
        )
        
        # Record memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_used = memory_mb - test_info["start_memory_mb"]
        
        if memory_mb > self.metrics["memory_usage_peak"]:
            self.metrics["memory_usage_peak"] = memory_mb
        
        # Track failures
        if not success:
            self.metrics["test_failures"] += 1
        
        # Track performance violations
        if duration > TEST_CONFIG["testing"]["performance_threshold_ms"] / 1000:
            self.metrics["performance_violations"] += 1
        
        # Track slowest tests
        test_record = {
            "name": test_info["test_name"],
            "duration": duration,
            "category": test_info["category"],
            "memory_used_mb": memory_used
        }
        
        self.metrics["slowest_tests"].append(test_record)
        self.metrics["slowest_tests"].sort(key=lambda x: x["duration"], reverse=True)
        self.metrics["slowest_tests"] = self.metrics["slowest_tests"][:10]  # Keep top 10
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test metrics report."""
        success_rate = (
            (self.metrics["test_runs"] - self.metrics["test_failures"]) / 
            max(self.metrics["test_runs"], 1) * 100
        )
        
        return {
            "summary": {
                "total_tests": self.metrics["test_runs"],
                "successful_tests": self.metrics["test_runs"] - self.metrics["test_failures"],
                "failed_tests": self.metrics["test_failures"],
                "success_rate_percent": success_rate,
                "total_duration_seconds": self.metrics["test_duration_total"],
                "average_duration_seconds": self.metrics["test_duration_avg"],
                "peak_memory_usage_mb": self.metrics["memory_usage_peak"],
                "performance_violations": self.metrics["performance_violations"]
            },
            "by_category": self.metrics["tests_by_category"],
            "slowest_tests": self.metrics["slowest_tests"],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance and reliability recommendations."""
        recommendations = []
        
        if self.metrics["test_failures"] > 0:
            failure_rate = self.metrics["test_failures"] / self.metrics["test_runs"] * 100
            if failure_rate > 5:
                recommendations.append(
                    f"High failure rate ({failure_rate:.1f}%). Review failing tests for flakiness."
                )
        
        if self.metrics["test_duration_avg"] > 1.0:
            recommendations.append(
                f"Average test duration is {self.metrics['test_duration_avg']:.2f}s. "
                "Consider optimizing slow tests or using parallel execution."
            )
        
        if self.metrics["memory_usage_peak"] > TEST_CONFIG["testing"]["memory_limit_mb"]:
            recommendations.append(
                f"Peak memory usage ({self.metrics['memory_usage_peak']:.1f}MB) exceeds limit "
                f"({TEST_CONFIG['testing']['memory_limit_mb']}MB). Review memory-intensive tests."
            )
        
        if self.metrics["performance_violations"] > 0:
            recommendations.append(
                f"{self.metrics['performance_violations']} tests exceeded performance threshold. "
                "Review and optimize slow tests."
            )
        
        return recommendations


# Pytest fixtures for enhanced testing
@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment for entire session."""
    env_manager = TestEnvironmentManager()
    env_manager.setup_test_environment()
    
    yield env_manager
    
    env_manager.teardown_test_environment()


@pytest.fixture(scope="function")
def test_data_manager():
    """Provide test data manager for each test."""
    manager = TestDataManager()
    
    yield manager
    
    manager.cleanup()


@pytest.fixture(scope="session")
def test_metrics():
    """Collect test metrics throughout session."""
    collector = TestMetricsCollector()
    
    yield collector
    
    # Generate final report
    report = collector.generate_report()
    
    # Save report
    report_path = Path("tests/reports/test_metrics_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Test metrics report saved to {report_path}")
    print(f"✅ Success rate: {report['summary']['success_rate_percent']:.1f}%")
    print(f"⏱️  Average duration: {report['summary']['average_duration_seconds']:.3f}s")
    print(f"🧠 Peak memory: {report['summary']['peak_memory_usage_mb']:.1f}MB")


@pytest.fixture
def mock_ai_services():
    """Mock AI services for isolated testing."""
    mocks = {}
    
    # Mock OpenAI
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mock AI response"))]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        mocks['openai'] = mock_client
    
    # Mock Google AI
    with patch('google.generativeai.GenerativeModel') as mock_google:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="Mock Google AI response")
        mock_google.return_value = mock_model
        mocks['google_ai'] = mock_model
    
    yield mocks


@pytest.fixture
def mock_database():
    """Mock database for isolated testing."""
    with patch('asyncpg.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


# Custom pytest markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "security: Security tests",
    "slow: Slow tests (> 5 seconds)",
    "fast: Fast tests (< 1 second)",
    "database: Tests requiring database",
    "ai_service: Tests requiring AI services",
    "network: Tests requiring network access"
]