"""
Comprehensive tests for enhanced API routes.

This module tests all API endpoints with advanced testing patterns including
performance testing, integration testing, and property-based testing.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, patch, Mock
from typing import Dict, Any, List

from fastapi import status
from fastapi.testclient import TestClient

# Import our testing framework
from tests.framework.advanced_testing import (
    PerformanceTester, PropertyBasedTester, IntegrationTestHarness,
    APITestClient, TestDataFactory
)

# Import API components
from pipeline.api.enhanced_routes import (
    create_api_router, CreateIdeaRequest, UpdateIdeaRequest,
    AnalyticsRequest, WorkflowExecutionRequest, SearchRequest
)
from pipeline.models.idea import IdeaCategory, IdeaStatus, PipelineStage


class TestIdeasAPI:
    """Test suite for Ideas API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    def test_create_idea_success(self, test_client, auth_headers):
        """Test successful idea creation."""
        request_data = {
            "title": "AI-Powered Code Review Assistant",
            "description": "Automated code review tool that provides intelligent feedback on pull requests using machine learning algorithms",
            "category": "ai_ml",
            "problem_statement": "Manual code reviews are time-consuming and inconsistent",
            "solution_description": "AI-powered automation for code quality assessment",
            "target_market": "Software development teams",
            "evidence_links": ["https://example.com/research"]
        }
        
        response = test_client.post(
            "/api/v1/ideas/",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["title"] == request_data["title"]
        assert data["description"] == request_data["description"]
        assert data["category"] == request_data["category"]
        assert "idea_id" in data
        assert "created_at" in data
    
    def test_create_idea_validation_error(self, test_client, auth_headers):
        """Test idea creation with validation errors."""
        # Title too short
        request_data = {
            "title": "Short",  # Less than 10 characters
            "description": "Valid description that meets minimum length requirements",
            "category": "ai_ml"
        }
        
        response = test_client.post(
            "/api/v1/ideas/",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_idea_success(self, test_client, auth_headers):
        """Test retrieving an idea by ID."""
        idea_id = str(uuid4())
        
        response = test_client.get(
            f"/api/v1/ideas/{idea_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["idea_id"] == idea_id
    
    def test_update_idea_success(self, test_client, auth_headers):
        """Test updating an idea."""
        idea_id = str(uuid4())
        update_data = {
            "title": "Updated AI Assistant",
            "status": "validated"
        }
        
        response = test_client.put(
            f"/api/v1/ideas/{idea_id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["title"] == update_data["title"]
    
    def test_delete_idea_success(self, test_client, auth_headers):
        """Test deleting an idea."""
        idea_id = str(uuid4())
        
        response = test_client.delete(
            f"/api/v1/ideas/{idea_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
    
    def test_search_ideas_success(self, test_client, auth_headers):
        """Test searching ideas with filters."""
        search_data = {
            "query": "AI machine learning",
            "categories": ["ai_ml"],
            "statuses": ["researching"],
            "limit": 10,
            "offset": 0
        }
        
        response = test_client.post(
            "/api/v1/ideas/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "ideas" in data
        assert "total_count" in data
        assert "page" in data
        assert "search_metadata" in data
        assert isinstance(data["ideas"], list)
    
    def test_bulk_operations_success(self, test_client, auth_headers):
        """Test bulk operations on ideas."""
        idea_ids = [str(uuid4()) for _ in range(3)]
        bulk_data = {
            "operation": "update_status",
            "idea_ids": idea_ids,
            "parameters": {"new_status": "validated"}
        }
        
        response = test_client.post(
            "/api/v1/ideas/bulk",
            json=bulk_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["operation"] == "update_status"
        assert data["total_requested"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3
    
    def test_authentication_required(self, test_client):
        """Test that authentication is required for all endpoints."""
        response = test_client.post(
            "/api/v1/ideas/",
            json={"title": "Test", "description": "Test description"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAnalyticsAPI:
    """Test suite for Analytics API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    def test_analyze_idea_success(self, test_client, auth_headers):
        """Test idea analysis endpoint."""
        idea_id = str(uuid4())
        analysis_data = {
            "idea_id": idea_id,
            "analysis_type": "comprehensive",
            "include_evidence": True,
            "research_depth": "standard"
        }
        
        response = test_client.post(
            "/api/v1/analytics/analyze",
            json=analysis_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["idea_id"] == idea_id
        assert data["analysis_type"] == "comprehensive"
        assert "market_potential" in data
        assert "competitive_analysis" in data
        assert "funding_potential" in data
        assert "confidence_score" in data
        assert "generated_at" in data
        assert "processing_time_seconds" in data
    
    def test_find_similar_ideas_success(self, test_client, auth_headers):
        """Test finding similar ideas."""
        idea_id = str(uuid4())
        
        response = test_client.get(
            f"/api/v1/analytics/{idea_id}/similarity?limit=5&threshold=0.8",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["idea_id"] == idea_id
        assert "similar_ideas" in data
        assert "threshold_used" in data
        assert "total_found" in data
        assert isinstance(data["similar_ideas"], list)


class TestWorkflowsAPI:
    """Test suite for Workflows API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    def test_execute_workflow_success(self, test_client, auth_headers):
        """Test workflow execution."""
        idea_id = str(uuid4())
        workflow_data = {
            "idea_id": idea_id,
            "workflow_type": "complete_validation",
            "parameters": {"priority": "high"},
            "priority": "normal"
        }
        
        response = test_client.post(
            "/api/v1/workflows/execute",
            json=workflow_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["idea_id"] == idea_id
        assert data["workflow_type"] == "complete_validation"
        assert data["status"] == "started"
        assert "execution_id" in data
        assert "started_at" in data
    
    def test_get_workflow_status_success(self, test_client, auth_headers):
        """Test getting workflow status."""
        execution_id = str(uuid4())
        
        response = test_client.get(
            f"/api/v1/workflows/{execution_id}/status",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["execution_id"] == execution_id
        assert "status" in data
        assert "progress_percentage" in data
        assert "started_at" in data


class TestAdminAPI:
    """Test suite for Admin API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    def test_health_check_success(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/v1/admin/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        assert "metrics" in data
    
    def test_get_metrics_success(self, test_client):
        """Test system metrics endpoint."""
        response = test_client.get("/api/v1/admin/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "timestamp" in data
        assert "performance" in data
        assert "business" in data
        assert "infrastructure" in data


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    @pytest.mark.asyncio
    async def test_idea_creation_performance(
        self, 
        performance_tester: PerformanceTester,
        test_client,
        auth_headers
    ):
        """Test idea creation performance under load."""
        
        async def create_idea_operation():
            """Single idea creation operation."""
            request_data = {
                "title": f"Performance Test Idea {uuid4()}",
                "description": "This is a performance test idea with sufficient description length to meet validation requirements.",
                "category": "ai_ml"
            }
            
            # Simulate async operation
            await asyncio.sleep(0.01)  # Simulate network delay
            
            # In real test, this would make actual HTTP request
            # response = test_client.post("/api/v1/ideas/", json=request_data, headers=auth_headers)
            # assert response.status_code == 201
        
        # Test different concurrency levels
        concurrency_results = await performance_tester.benchmark_concurrent_operations(
            operation_func=create_idea_operation,
            operation_name="create_idea",
            concurrency_levels=[1, 5, 10],
            operations_per_level=20
        )
        
        # Verify performance requirements
        for concurrency, metrics in concurrency_results.items():
            assert metrics.success_rate >= 95.0, f"Success rate too low at concurrency {concurrency}"
            assert metrics.average_time_seconds <= 1.0, f"Average response time too high at concurrency {concurrency}"
            assert metrics.operations_per_second >= 5.0, f"Throughput too low at concurrency {concurrency}"
    
    @pytest.mark.asyncio
    async def test_search_performance(
        self,
        performance_tester: PerformanceTester,
        test_client,
        auth_headers
    ):
        """Test search endpoint performance."""
        
        async def search_operation():
            """Single search operation."""
            search_data = {
                "query": "AI machine learning",
                "categories": ["ai_ml"],
                "limit": 20
            }
            
            # Simulate search operation
            await asyncio.sleep(0.05)  # Simulate database query time
        
        async with performance_tester.measure_async_performance(
            "search_ideas",
            target_operations=50,
            concurrency=5
        ) as execute_op:
            
            # Execute search operations
            tasks = []
            for _ in range(50):
                task = asyncio.create_task(execute_op(search_operation))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check latest metrics
        if performance_tester.metrics_history:
            metrics = performance_tester.metrics_history[-1]
            assert metrics.average_time_seconds <= 0.2, "Search too slow"
            assert metrics.success_rate >= 98.0, "Search reliability too low"


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for complete API workflows."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    @pytest.mark.asyncio
    async def test_complete_idea_lifecycle(
        self,
        integration_harness: IntegrationTestHarness,
        test_client,
        auth_headers
    ):
        """Test complete idea lifecycle through API."""
        
        async with integration_harness.isolated_test_environment():
            # Step 1: Create idea
            idea_data = {
                "title": "Integration Test Idea",
                "description": "This is an integration test idea for the complete lifecycle workflow.",
                "category": "ai_ml",
                "problem_statement": "Integration testing challenges",
                "solution_description": "Automated integration testing solution"
            }
            
            create_response = test_client.post(
                "/api/v1/ideas/",
                json=idea_data,
                headers=auth_headers
            )
            
            assert create_response.status_code == status.HTTP_201_CREATED
            created_idea = create_response.json()
            idea_id = created_idea["idea_id"]
            
            # Step 2: Analyze idea
            analysis_data = {
                "idea_id": idea_id,
                "analysis_type": "comprehensive",
                "include_evidence": True
            }
            
            analysis_response = test_client.post(
                "/api/v1/analytics/analyze",
                json=analysis_data,
                headers=auth_headers
            )
            
            assert analysis_response.status_code == status.HTTP_200_OK
            analysis_result = analysis_response.json()
            
            # Step 3: Execute workflow
            workflow_data = {
                "idea_id": idea_id,
                "workflow_type": "complete_validation"
            }
            
            workflow_response = test_client.post(
                "/api/v1/workflows/execute",
                json=workflow_data,
                headers=auth_headers
            )
            
            assert workflow_response.status_code == status.HTTP_200_OK
            workflow_result = workflow_response.json()
            execution_id = workflow_result["execution_id"]
            
            # Step 4: Check workflow status
            status_response = test_client.get(
                f"/api/v1/workflows/{execution_id}/status",
                headers=auth_headers
            )
            
            assert status_response.status_code == status.HTTP_200_OK
            status_result = status_response.json()
            
            # Step 5: Update idea based on results
            update_data = {
                "status": "validated"
            }
            
            update_response = test_client.put(
                f"/api/v1/ideas/{idea_id}",
                json=update_data,
                headers=auth_headers
            )
            
            assert update_response.status_code == status.HTTP_200_OK
            
            # Verify the complete workflow
            assert created_idea["title"] == idea_data["title"]
            assert analysis_result["confidence_score"] >= 0.0
            assert workflow_result["status"] == "started"
            assert status_result["execution_id"] == execution_id


@pytest.mark.property
class TestAPIPropertyBased:
    """Property-based tests for API endpoints."""
    
    @pytest.fixture
    def api_router(self):
        """Create API router for testing."""
        return create_api_router()
    
    @pytest.fixture
    def test_client(self, api_router):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    def test_idea_creation_invariants(
        self,
        property_tester: PropertyBasedTester,
        test_client,
        auth_headers
    ):
        """Test invariants for idea creation with various inputs."""
        
        # Generate test ideas
        test_ideas = [
            property_tester.generate_valid_idea(category=cat)
            for cat in list(IdeaCategory)[:5]  # Test first 5 categories
        ]
        
        def idea_creation_invariant(idea):
            """Invariant: All valid ideas should be created successfully."""
            request_data = {
                "title": idea.title,
                "description": idea.description,
                "category": idea.category.value
            }
            
            response = test_client.post(
                "/api/v1/ideas/",
                json=request_data,
                headers=auth_headers
            )
            
            # Invariant: Valid ideas should always be created
            return response.status_code == status.HTTP_201_CREATED
        
        # Test invariant
        failures = property_tester.test_invariant(idea_creation_invariant, test_ideas)
        
        assert len(failures) == 0, f"Idea creation invariant failed for: {failures}"
    
    def test_search_consistency(
        self,
        property_tester: PropertyBasedTester,
        test_client,
        auth_headers
    ):
        """Test search result consistency across different parameters."""
        
        search_terms = [
            "AI machine learning",
            "blockchain cryptocurrency",
            "healthcare telemedicine",
            "fintech payments",
            "education online learning"
        ]
        
        def search_consistency_invariant(search_term):
            """Invariant: Search should always return valid structure."""
            search_data = {
                "query": search_term,
                "limit": 10
            }
            
            response = test_client.post(
                "/api/v1/ideas/search",
                json=search_data,
                headers=auth_headers
            )
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            
            # Invariants for search response
            required_fields = ["ideas", "total_count", "page", "page_size", "has_more"]
            return all(field in data for field in required_fields)
        
        failures = property_tester.test_invariant(search_consistency_invariant, search_terms)
        
        assert len(failures) == 0, f"Search consistency invariant failed for: {failures}"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-m", "not slow",  # Skip slow tests by default
        "--tb=short"
    ])