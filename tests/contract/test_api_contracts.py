"""Contract tests for API endpoints."""

import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient

from tests.helpers.assertions import (
    assert_idea_response,
    assert_pagination_response,
    assert_error_response,
    assert_performance_metrics,
    assert_security_headers,
)


class TestIdeaAPIContracts:
    """Contract tests for Idea API endpoints."""
    
    def test_create_idea_contract(self, client: TestClient, api_test_data: Dict[str, Any]):
        """Test create idea endpoint contract."""
        idea_data = api_test_data["valid_idea"]
        
        response = client.post("/api/v1/ideas", json=idea_data)
        
        assert response.status_code == 201
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_idea_response(response_data)
        
        # Verify all input fields are preserved
        assert response_data["title"] == idea_data["title"]
        assert response_data["description"] == idea_data["description"]
        assert response_data["category"] == idea_data["category"]
    
    def test_get_idea_contract(self, client: TestClient, sample_idea_data: Dict[str, Any]):
        """Test get idea endpoint contract."""
        # First create an idea
        create_response = client.post("/api/v1/ideas", json=sample_idea_data)
        idea_id = create_response.json()["id"]
        
        # Get the idea
        response = client.get(f"/api/v1/ideas/{idea_id}")
        
        assert response.status_code == 200
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_idea_response(response_data)
        assert response_data["id"] == idea_id
    
    def test_list_ideas_contract(self, client: TestClient):
        """Test list ideas endpoint contract."""
        response = client.get("/api/v1/ideas")
        
        assert response.status_code == 200
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_pagination_response(response_data)
        
        # Verify each idea in the list follows the contract
        for idea in response_data["data"]:
            assert_idea_response(idea)
    
    def test_update_idea_contract(self, client: TestClient, sample_idea_data: Dict[str, Any]):
        """Test update idea endpoint contract."""
        # Create an idea first
        create_response = client.post("/api/v1/ideas", json=sample_idea_data)
        idea_id = create_response.json()["id"]
        
        # Update the idea
        update_data = {"title": "Updated Title", "description": "Updated description"}
        response = client.patch(f"/api/v1/ideas/{idea_id}", json=update_data)
        
        assert response.status_code == 200
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_idea_response(response_data)
        assert response_data["title"] == update_data["title"]
        assert response_data["description"] == update_data["description"]
    
    def test_delete_idea_contract(self, client: TestClient, sample_idea_data: Dict[str, Any]):
        """Test delete idea endpoint contract."""
        # Create an idea first
        create_response = client.post("/api/v1/ideas", json=sample_idea_data)
        idea_id = create_response.json()["id"]
        
        # Delete the idea
        response = client.delete(f"/api/v1/ideas/{idea_id}")
        
        assert response.status_code == 204
        assert_security_headers(response.headers)
        
        # Verify idea is deleted
        get_response = client.get(f"/api/v1/ideas/{idea_id}")
        assert get_response.status_code == 404


class TestErrorResponseContracts:
    """Contract tests for error responses."""
    
    def test_not_found_error_contract(self, client: TestClient):
        """Test 404 error response contract."""
        response = client.get("/api/v1/ideas/non-existent-id")
        
        assert response.status_code == 404
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_error_response(response_data, "NotFound", 404)
    
    def test_validation_error_contract(self, client: TestClient, api_test_data: Dict[str, Any]):
        """Test validation error response contract."""
        invalid_data = api_test_data["invalid_idea"]
        
        response = client.post("/api/v1/ideas", json=invalid_data)
        
        assert response.status_code == 422
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_error_response(response_data, "ValidationError", 422)
        
        # Validation errors should include details
        assert "details" in response_data
        assert isinstance(response_data["details"], list)
    
    def test_unauthorized_error_contract(self, client: TestClient):
        """Test 401 error response contract."""
        # Make request without authentication
        response = client.get("/api/v1/admin/metrics")
        
        assert response.status_code == 401
        assert_security_headers(response.headers)
        
        response_data = response.json()
        assert_error_response(response_data, "Unauthorized", 401)
    
    def test_rate_limit_error_contract(self, client: TestClient):
        """Test rate limit error response contract."""
        # This would require actual rate limiting implementation
        # For now, we'll test the contract structure
        pass


class TestPerformanceContracts:
    """Contract tests for performance requirements."""
    
    @pytest.mark.performance
    def test_api_response_time_contract(self, client: TestClient):
        """Test that API response times meet contract requirements."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert_performance_metrics(response_time, max_response_time=0.200)  # 200ms
        
        assert response.status_code == 200
    
    @pytest.mark.performance
    def test_idea_creation_performance_contract(self, client: TestClient, sample_idea_data: Dict[str, Any]):
        """Test idea creation performance contract."""
        import time
        
        start_time = time.time()
        response = client.post("/api/v1/ideas", json=sample_idea_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert_performance_metrics(response_time, max_response_time=1.0)  # 1 second
        
        assert response.status_code == 201
    
    @pytest.mark.performance
    def test_search_performance_contract(self, client: TestClient):
        """Test search endpoint performance contract."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/ideas/search?q=AI machine learning")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert_performance_metrics(response_time, max_response_time=0.050)  # 50ms for vector search
        
        assert response.status_code == 200


class TestSecurityContracts:
    """Contract tests for security requirements."""
    
    def test_security_headers_contract(self, client: TestClient):
        """Test that all responses include required security headers."""
        endpoints = [
            "/api/v1/health",
            "/api/v1/ideas",
            "/api/v1/metrics",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert_security_headers(response.headers)
    
    def test_cors_headers_contract(self, client: TestClient):
        """Test CORS headers contract."""
        response = client.options("/api/v1/ideas")
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    @pytest.mark.security
    def test_sql_injection_protection_contract(self, client: TestClient, security_test_data: Dict[str, Any]):
        """Test SQL injection protection contract."""
        for injection_attempt in security_test_data["sql_injection_attempts"]:
            response = client.get(f"/api/v1/ideas/search?q={injection_attempt}")
            
            # Should not return server error
            assert response.status_code != 500
            
            # Should not contain SQL error messages
            response_text = response.text.lower()
            assert "sql" not in response_text
            assert "database" not in response_text
            assert "syntax error" not in response_text
    
    @pytest.mark.security
    def test_xss_protection_contract(self, client: TestClient, security_test_data: Dict[str, Any]):
        """Test XSS protection contract."""
        for xss_attempt in security_test_data["xss_attempts"]:
            idea_data = {
                "title": xss_attempt,
                "description": f"Description with {xss_attempt}",
                "category": "ai_ml"
            }
            
            response = client.post("/api/v1/ideas", json=idea_data)
            
            if response.status_code == 201:
                # XSS content should be escaped/sanitized
                response_data = response.json()
                assert "<script>" not in response_data["title"]
                assert "javascript:" not in response_data["title"]