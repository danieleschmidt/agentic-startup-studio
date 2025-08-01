"""Test decorators and utilities for enhancing test functionality."""

import functools
import time
import asyncio
from typing import Any, Callable, Dict, Optional
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime, timezone


def timeout(seconds: float = 5.0):
    """Decorator to set timeout for test functions."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > seconds:
                    pytest.fail(f"Test timed out after {elapsed:.2f}s (limit: {seconds}s)")
                return result
            return sync_wrapper
    return decorator


def performance_test(
    max_duration: float = 1.0,
    max_memory_mb: Optional[float] = None,
    profile: bool = False
):
    """Decorator for performance testing with duration and memory constraints."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Record start time
            start_time = time.time()
            
            if profile:
                import cProfile
                import pstats
                profiler = cProfile.Profile()
                profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Calculate duration
                duration = time.time() - start_time
                
                # Check memory usage
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
                
                if profile:
                    profiler.disable()
                    stats = pstats.Stats(profiler)
                    stats.sort_stats('cumulative')
                    print(f"\n=== Performance Profile for {func.__name__} ===")
                    stats.print_stats(10)  # Top 10 functions
                
                # Assert performance constraints
                if duration > max_duration:
                    pytest.fail(
                        f"Performance test failed: {func.__name__} took {duration:.3f}s "
                        f"(max: {max_duration}s)"
                    )
                
                if max_memory_mb and memory_used > max_memory_mb:
                    pytest.fail(
                        f"Memory test failed: {func.__name__} used {memory_used:.2f}MB "
                        f"(max: {max_memory_mb}MB)"
                    )
                
                # Add performance metadata to test result
                if hasattr(result, '__dict__'):
                    result.__dict__['_performance_data'] = {
                        'duration': duration,
                        'memory_used_mb': memory_used,
                        'measured_at': datetime.now(timezone.utc).isoformat()
                    }
            
            return result
        return wrapper
    return decorator


def mock_external_services(*service_names: str):
    """Decorator to mock external services for isolated testing."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            patches = []
            mocks = {}
            
            try:
                # Create mocks for each service
                for service_name in service_names:
                    if service_name == "openai":
                        mock_obj = MagicMock()
                        mock_obj.chat.completions.create.return_value = MagicMock(
                            choices=[MagicMock(message=MagicMock(content="Mock AI response"))],
                            usage=MagicMock(total_tokens=100)
                        )
                        patch_obj = patch('openai.OpenAI', return_value=mock_obj)
                    
                    elif service_name == "database":
                        mock_obj = MagicMock()
                        mock_obj.execute.return_value = MagicMock(fetchall=lambda: [])
                        patch_obj = patch('sqlalchemy.create_engine', return_value=mock_obj)
                    
                    elif service_name == "redis":
                        mock_obj = MagicMock()
                        mock_obj.get.return_value = None
                        mock_obj.set.return_value = True
                        patch_obj = patch('redis.Redis', return_value=mock_obj)
                    
                    elif service_name == "http":
                        mock_obj = MagicMock()
                        mock_obj.get.return_value = MagicMock(
                            status_code=200,
                            json=lambda: {"status": "success", "data": {}}
                        )
                        patch_obj = patch('requests.Session', return_value=mock_obj)
                    
                    else:
                        # Generic mock
                        patch_obj = patch(service_name, MagicMock())
                        mock_obj = patch_obj.start()
                    
                    patches.append(patch_obj)
                    mocks[service_name] = mock_obj
                    patch_obj.start()
                
                # Add mocks to kwargs so test can access them
                kwargs['_mocks'] = mocks
                
                return func(*args, **kwargs)
            
            finally:
                # Stop all patches
                for patch_obj in patches:
                    patch_obj.stop()
        
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 0.1, exceptions: tuple = (Exception,)):
    """Decorator to retry test functions on specific exceptions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    break
            
            # If we get here, all retries failed
            pytest.fail(
                f"Test failed after {max_retries + 1} attempts. "
                f"Last exception: {type(last_exception).__name__}: {last_exception}"
            )
        
        return wrapper
    return decorator


def database_transaction(rollback: bool = True):
    """Decorator to wrap test in database transaction with optional rollback."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would integrate with your actual database connection
            # For now, it's a placeholder showing the pattern
            transaction_started = False
            
            try:
                # Begin transaction
                # db.begin_transaction()
                transaction_started = True
                
                result = func(*args, **kwargs)
                
                if rollback and transaction_started:
                    # Rollback transaction
                    # db.rollback_transaction()
                    pass
                else:
                    # Commit transaction
                    # db.commit_transaction()
                    pass
                
                return result
            
            except Exception as e:
                if transaction_started:
                    # Rollback on error
                    # db.rollback_transaction()
                    pass
                raise e
        
        return wrapper
    return decorator


def freeze_time(frozen_time: str):
    """Decorator to freeze time for consistent testing."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.fromisoformat(frozen_time)
                mock_datetime.utcnow.return_value = datetime.fromisoformat(frozen_time)
                return func(*args, **kwargs)
        return wrapper
    return decorator


def test_category(*categories: str):
    """Decorator to categorize tests for selective running."""
    def decorator(func: Callable) -> Callable:
        # Add pytest markers for each category
        for category in categories:
            func = pytest.mark.__getattr__(category)(func)
        
        # Store categories as metadata
        func._test_categories = list(categories)
        return func
    return decorator


# Composite decorators for common test patterns
def integration_test(timeout_seconds: float = 30.0):
    """Composite decorator for integration tests."""
    def decorator(func: Callable) -> Callable:
        func = test_category("integration")(func)
        func = timeout(timeout_seconds)(func)
        func = retry_on_failure(max_retries=2)(func)
        return func
    return decorator


def unit_test(performance_check: bool = False):
    """Composite decorator for unit tests."""
    def decorator(func: Callable) -> Callable:
        func = test_category("unit")(func)
        func = timeout(5.0)(func)
        
        if performance_check:
            func = performance_test(max_duration=0.1)(func)
        
        return func
    return decorator


def e2e_test(timeout_seconds: float = 60.0):
    """Composite decorator for end-to-end tests."""
    def decorator(func: Callable) -> Callable:
        func = test_category("e2e", "slow")(func)
        func = timeout(timeout_seconds)(func)
        func = retry_on_failure(max_retries=1, delay=1.0)(func)
        return func
    return decorator