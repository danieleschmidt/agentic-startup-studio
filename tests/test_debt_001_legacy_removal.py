"""
Tests for DEBT-001: Legacy/Deprecated Code Removal

Validates that legacy code removal doesn't break existing functionality.
"""

import pytest
from unittest.mock import MagicMock


class TestDebt001LegacyRemoval:
    """Test that legacy code removal maintains functionality."""
    
    def test_circuit_breaker_functionality_intact(self):
        """Test that circuit breaker functionality works without deprecated alias."""
        from pipeline.infrastructure.circuit_breaker import circuit_breaker, CircuitBreaker
        
        # Test that the main circuit_breaker decorator still works
        @circuit_breaker("test_service", failure_threshold=3)
        def test_function():
            return "success"
        
        # Should work without throwing errors
        result = test_function()
        assert result == "success"
        
        # Test that CircuitBreaker class is still importable and functional
        cb = CircuitBreaker("test_cb")
        assert cb.name == "test_cb"
        assert cb.state.name == "CLOSED"  # Initial state
        
    def test_validation_result_is_success_method(self):
        """Test that ValidationResult.is_success() method works correctly."""
        from tests.framework.base import ValidationResult
        from tests.framework.config import ValidationArea, ValidationStatus
        
        # Test successful validation
        success_result = ValidationResult(
            validation_id="test_success",
            area=ValidationArea.DATA_FLOW,
            status=ValidationStatus.PASSED,
            message="Test passed"
        )
        
        assert success_result.is_success() is True
        
        # Test failed validation
        fail_result = ValidationResult(
            validation_id="test_fail",
            area=ValidationArea.DATA_FLOW,
            status=ValidationStatus.FAILED,
            message="Test failed"
        )
        
        assert fail_result.is_success() is False
        
    def test_semantic_scholar_adapter_creation(self):
        """Test that Semantic Scholar adapter creation works without global state."""
        from tools.semantic_scholar import _create_adapter, SemanticScholarAdapter
        
        # Test that the private helper function creates valid adapters
        adapter = _create_adapter()
        assert isinstance(adapter, SemanticScholarAdapter)
        assert adapter.config is not None
        
    @pytest.mark.asyncio
    async def test_semantic_scholar_async_interfaces_work(self):
        """Test that async interfaces work without global adapter."""
        from tools.semantic_scholar import search_papers_async, get_paper_details_async
        from unittest.mock import patch, AsyncMock
        
        # Mock the HTTP client to avoid actual API calls
        with patch('tools.semantic_scholar.SemanticScholarAdapter.search_papers') as mock_search:
            mock_search.return_value = [{"title": "Test Paper", "paperId": "123"}]
            
            # Test search function
            results = await search_papers_async("test query", limit=5)
            assert len(results) == 1
            assert results[0]["title"] == "Test Paper"
            mock_search.assert_called_once_with("test query", 5)
            
        with patch('tools.semantic_scholar.SemanticScholarAdapter.get_paper_details') as mock_details:
            mock_details.return_value = {"paperId": "123", "title": "Test Paper", "abstract": "Test abstract"}
            
            # Test paper details function
            details = await get_paper_details_async("123")
            assert details["paperId"] == "123"
            assert details["title"] == "Test Paper"
            mock_details.assert_called_once_with("123")
            
    def test_infrastructure_imports_still_work(self):
        """Test that infrastructure module imports work after removing deprecated alias."""
        # These imports should work without the deprecated circuit_breaker_decorator
        from pipeline.infrastructure import (
            CircuitBreaker,
            CircuitBreakerRegistry, 
            get_circuit_breaker_registry,
            create_api_circuit_breaker,
            create_database_circuit_breaker,
            create_llm_circuit_breaker
        )
        
        # Test that factory functions work
        api_cb = create_api_circuit_breaker("test_api")
        assert isinstance(api_cb, CircuitBreaker)
        assert api_cb.name == "test_api"
        
        db_cb = create_database_circuit_breaker("test_db")
        assert isinstance(db_cb, CircuitBreaker)
        assert db_cb.name == "test_db"
        
        llm_cb = create_llm_circuit_breaker("test_llm")
        assert isinstance(llm_cb, CircuitBreaker)
        assert llm_cb.name == "test_llm"
        
    def test_validation_engine_uses_is_success_method(self):
        """Test that validation engine correctly uses is_success() instead of passed property."""
        from tests.framework.validation_engine import ValidationEngine
        from tests.framework.base import ValidationResult
        from tests.framework.config import ValidationArea, ValidationStatus
        
        # Create test results
        results = {
            ValidationArea.DATA_FLOW: [
                ValidationResult(
                    validation_id="test1",
                    area=ValidationArea.DATA_FLOW,
                    status=ValidationStatus.PASSED,
                    message="Success"
                ),
                ValidationResult(
                    validation_id="test2", 
                    area=ValidationArea.DATA_FLOW,
                    status=ValidationStatus.FAILED,
                    message="Failed"
                )
            ]
        }
        
        # Test summary generation (uses is_success() internally)
        engine = ValidationEngine()
        summary = engine._generate_summary(results)
        
        assert summary["total_validations"] == 2
        assert summary["passed_validations"] == 1
        assert summary["failed_validations"] == 1
        assert summary["areas"][ValidationArea.DATA_FLOW.value]["passed"] == 1
        assert summary["areas"][ValidationArea.DATA_FLOW.value]["failed"] == 1


def test_debt_001_acceptance_criteria():
    """Test that DEBT-001 acceptance criteria are met."""
    
    # Check that deprecated patterns have been removed
    import inspect
    
    # 1. Circuit breaker alias should not exist
    from pipeline.infrastructure import circuit_breaker
    
    # The deprecated alias should not be accessible
    with pytest.raises(ImportError):
        from pipeline.infrastructure import circuit_breaker_decorator
        
    # 2. ValidationResult should not have passed property
    from tests.framework.base import ValidationResult
    
    # The property should not exist 
    assert not hasattr(ValidationResult, 'passed'), "Deprecated 'passed' property should be removed"
    
    # But is_success method should exist
    assert hasattr(ValidationResult, 'is_success'), "is_success method should exist"
    
    # 3. Semantic Scholar should not use global state
    from tools import semantic_scholar
    
    # Global adapter variable should not exist
    assert not hasattr(semantic_scholar, '_global_adapter'), "Global adapter should be removed"
    assert not hasattr(semantic_scholar, 'get_semantic_scholar_adapter'), "Global adapter getter should be removed"
    
    # But the helper function should exist
    assert hasattr(semantic_scholar, '_create_adapter'), "Private adapter creation helper should exist"
    
    print("✅ DEBT-001 acceptance criteria met:")
    print("  • Legacy circuit breaker alias removed")
    print("  • Backward compatibility property removed from ValidationResult")
    print("  • Global adapter pattern removed from Semantic Scholar")
    print("  • All functionality preserved with cleaner interfaces")


if __name__ == "__main__":
    # Run the acceptance criteria test directly
    test_debt_001_acceptance_criteria()
    print("🎉 DEBT-001 validation passed!")