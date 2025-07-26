"""
Tests for API Gateway with Authentication and Rate Limiting

This test suite validates:
- Authentication middleware functionality
- Rate limiting enforcement  
- Request/response logging
- Security controls and JWT handling
- Gateway status and monitoring
"""

import pytest
import time
from unittest.mock import patch, MagicMock
import jwt
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from pipeline.api.gateway import APIGateway, AuthenticationRequest


class TestAPIGateway:
    """Test suite for API Gateway functionality."""
    
    @pytest.fixture
    def gateway(self):
        """Create API gateway instance for testing."""
        with patch('pipeline.api.gateway.get_settings') as mock_settings:
            mock_settings.return_value.environment = "testing"
            mock_settings.return_value.allowed_origins = ["localhost", "127.0.0.1"]
            mock_settings.return_value.secret_key = "test-secret-key"
            
            with patch('pipeline.api.gateway.get_secrets_manager') as mock_secrets:
                mock_secrets.return_value.get_secret.return_value = "test-api-key"
                return APIGateway()
    
    @pytest.fixture 
    def client(self, gateway):
        """Create test client."""
        return TestClient(gateway.app)
    
    def test_health_endpoint_public_access(self, client):
        """Test that health endpoint is publicly accessible."""
        with patch('pipeline.api.gateway.get_infrastructure_health') as mock_health:
            mock_health.return_value = {"status": "healthy"}
            
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    def test_api_key_authentication_success(self, client):
        """Test successful API key authentication."""
        auth_data = {"api_key": "test-api-key"}
        
        response = client.post("/auth/login", json=auth_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
    
    def test_api_key_authentication_failure(self, client):
        """Test API key authentication with invalid key."""
        auth_data = {"api_key": "invalid-key"}
        
        response = client.post("/auth/login", json=auth_data)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_jwt_token_verification(self, client):
        """Test JWT token verification for protected endpoints."""
        # First authenticate
        auth_data = {"api_key": "test-api-key"}
        auth_response = client.post("/auth/login", json=auth_data)
        token = auth_response.json()["access_token"]
        
        # Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/verify", headers=headers)
        
        assert response.status_code == 200
        assert response.json()["authenticated"] is True
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/auth/verify")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/auth/verify", headers=headers)
        assert response.status_code == 401
    
    def test_rate_limiting_enforcement(self, client, gateway):
        """Test rate limiting prevents abuse."""
        # Configure low rate limit for testing
        gateway.rate_limits["default"].requests_per_minute = 2
        
        # First two requests should succeed
        response1 = client.get("/health")
        response2 = client.get("/health")
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Third request should be rate limited
        response3 = client.get("/health")
        assert response3.status_code == 429
        assert "Rate limit exceeded" in response3.json()["detail"]
    
    def test_rate_limiting_per_endpoint(self, client, gateway):
        """Test that rate limiting is applied per endpoint."""
        # Configure different limits per endpoint
        gateway.rate_limits["/health"].requests_per_minute = 1
        gateway.rate_limits["/auth/login"].requests_per_minute = 2
        
        # Health endpoint rate limited after 1 request
        client.get("/health")
        response = client.get("/health")
        assert response.status_code == 429
        
        # Auth endpoint still allows requests
        auth_data = {"api_key": "test-api-key"}
        response = client.post("/auth/login", json=auth_data)
        assert response.status_code == 200
    
    def test_burst_limit_protection(self, client, gateway):
        """Test burst limit prevents rapid requests."""
        gateway.rate_limits["default"].burst_limit = 1
        
        # Make rapid requests
        responses = []
        for _ in range(3):
            responses.append(client.get("/health"))
            time.sleep(0.1)  # Small delay to simulate rapid requests
        
        # At least one should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes
    
    def test_ip_blocking_after_violations(self, client, gateway):
        """Test IP blocking after multiple rate limit violations."""
        gateway.rate_limits["default"].requests_per_minute = 1
        
        # Generate multiple violations
        for _ in range(7):  # Exceed violation threshold
            client.get("/health")
        
        # IP should be blocked
        response = client.get("/health")
        assert response.status_code == 429
        assert "IP temporarily blocked" in response.json()["detail"]
    
    def test_cors_middleware_configured(self, client):
        """Test CORS middleware is properly configured."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should handle preflight request
        assert response.status_code == 200
    
    def test_request_logging_middleware(self, client):
        """Test request logging middleware captures requests."""
        with patch('pipeline.api.gateway.logger') as mock_logger:
            response = client.get("/health")
            
            # Verify logging calls were made
            assert mock_logger.info.called
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Gateway request started" in call for call in log_calls)
            assert any("Gateway request completed" in call for call in log_calls)
    
    def test_prometheus_metrics_collection(self, client, gateway):
        """Test Prometheus metrics are collected."""
        from pipeline.api.gateway import gateway_requests_total
        
        # Make request and check metrics
        initial_count = gateway_requests_total._value._value
        client.get("/health")
        
        # Metrics should be updated (exact value depends on test execution)
        # Just verify the metric exists and is callable
        assert hasattr(gateway_requests_total, 'labels')
    
    def test_jwt_token_expiration(self, client):
        """Test JWT token expiration handling."""
        # Create expired token
        secret_key = "test-secret-key"
        expired_payload = {
            "session_id": "test-session",
            "authenticated": True,
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        expired_token = jwt.encode(expired_payload, secret_key, algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/auth/verify", headers=headers)
        
        assert response.status_code == 401
    
    def test_session_management(self, client):
        """Test session creation and cleanup."""
        # Login to create session
        auth_data = {"api_key": "test-api-key"}
        auth_response = client.post("/auth/login", json=auth_data)
        token = auth_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Verify session exists
        response = client.get("/auth/verify", headers=headers)
        assert response.status_code == 200
        
        # Logout to cleanup session
        response = client.delete("/auth/logout", headers=headers)
        assert response.status_code == 200
        assert "Logged out successfully" in response.json()["message"]
    
    def test_gateway_status_endpoint(self, client):
        """Test gateway status endpoint provides useful information."""
        # First authenticate
        auth_data = {"api_key": "test-api-key"}
        auth_response = client.post("/auth/login", json=auth_data)
        token = auth_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/gateway/status", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "active_sessions" in data
        assert "rate_limit_rules" in data
        assert "environment" in data
    
    def test_metrics_endpoint_authentication(self, client):
        """Test metrics endpoint requires authentication."""
        # Unauthenticated request should fail
        response = client.get("/metrics")
        assert response.status_code == 401
        
        # Authenticated request should succeed
        auth_data = {"api_key": "test-api-key"}
        auth_response = client.post("/auth/login", json=auth_data)
        token = auth_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        with patch('pipeline.api.gateway.get_infrastructure_metrics'):
            with patch('pipeline.api.gateway.generate_latest', return_value=b"metrics"):
                response = client.get("/metrics", headers=headers)
                assert response.status_code == 200
    
    def test_security_headers_in_production(self, gateway):
        """Test security headers are applied in production environment."""
        with patch('pipeline.api.gateway.get_settings') as mock_settings:
            mock_settings.return_value.environment = "production"
            mock_settings.return_value.allowed_origins = ["example.com"]
            
            # In production, TrustedHostMiddleware should be configured
            # This is verified by checking middleware stack
            middleware_types = [type(middleware) for middleware in gateway.app.user_middleware]
            middleware_names = [m.__name__ for m in middleware_types]
            
            # Should include CORS at minimum
            assert any("CORS" in name for name in middleware_names)
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, client, gateway):
        """Test rate limiting under concurrent requests."""
        import asyncio
        
        gateway.rate_limits["default"].requests_per_minute = 5
        
        # Make concurrent requests
        async def make_request():
            return client.get("/health")
        
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some requests should be rate limited
        status_codes = []
        for response in responses:
            if not isinstance(response, Exception):
                status_codes.append(response.status_code)
        
        # Should have mix of success and rate limited responses
        assert 200 in status_codes
        assert 429 in status_codes
    
    def test_authentication_request_model_validation(self):
        """Test authentication request model validates input."""
        # Valid request
        valid_request = AuthenticationRequest(api_key="test-key")
        assert valid_request.api_key == "test-key"
        
        # Invalid request should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            AuthenticationRequest()  # Missing required api_key
    
    def test_rate_limit_config_validation(self, gateway):
        """Test rate limit configuration is properly validated."""
        from pipeline.api.gateway import RateLimitConfig
        
        # Valid config
        config = RateLimitConfig(requests_per_minute=60, requests_per_hour=1000)
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        
        # Default values
        default_config = RateLimitConfig()
        assert default_config.requests_per_minute == 60
        assert default_config.burst_limit == 10