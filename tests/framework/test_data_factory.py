"""Test data factory for generating consistent test data across the test suite."""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import uuid4
from faker import Faker
import random

from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    ValidationResult, DuplicateCheckResult
)
from pipeline.services.budget_sentinel import BudgetCategory
from pipeline.services.pitch_deck_generator import InvestorType

fake = Faker()


class TestDataFactory:
    """Factory for creating consistent test data objects."""

    @staticmethod
    def create_idea_draft(
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[IdeaCategory] = None,
        **kwargs
    ) -> IdeaDraft:
        """Create a test IdeaDraft object."""
        return IdeaDraft(
            title=title or fake.catch_phrase(),
            description=description or fake.text(max_nb_chars=500),
            category=category or random.choice(list(IdeaCategory)),
            source="test_factory",
            raw_input=kwargs.get("raw_input", fake.text(max_nb_chars=200)),
            metadata=kwargs.get("metadata", {"test": True}),
            created_at=kwargs.get("created_at", datetime.now(timezone.utc))
        )

    @staticmethod
    def create_idea(
        idea_id: Optional[str] = None,
        status: Optional[IdeaStatus] = None,
        stage: Optional[PipelineStage] = None,
        **kwargs
    ) -> Idea:
        """Create a test Idea object."""
        return Idea(
            id=idea_id or str(uuid4()),
            title=kwargs.get("title", fake.catch_phrase()),
            description=kwargs.get("description", fake.text(max_nb_chars=500)),
            category=kwargs.get("category", random.choice(list(IdeaCategory))),
            status=status or IdeaStatus.DRAFT,
            stage=stage or PipelineStage.INGESTION,
            source="test_factory",
            confidence_score=kwargs.get("confidence_score", random.uniform(0.5, 1.0)),
            quality_metrics=kwargs.get("quality_metrics", {
                "clarity": random.uniform(0.6, 1.0),
                "specificity": random.uniform(0.6, 1.0),
                "market_potential": random.uniform(0.5, 1.0)
            }),
            processing_metadata=kwargs.get("processing_metadata", {
                "test_mode": True,
                "created_by": "test_factory"
            }),
            created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
            updated_at=kwargs.get("updated_at", datetime.now(timezone.utc))
        )

    @staticmethod
    def create_validation_result(
        is_valid: bool = True,
        confidence: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Create a test ValidationResult object."""
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence or random.uniform(0.7, 1.0) if is_valid else random.uniform(0.1, 0.4),
            validation_errors=kwargs.get("validation_errors", [] if is_valid else [fake.sentence()]),
            quality_score=kwargs.get("quality_score", random.uniform(0.7, 1.0) if is_valid else random.uniform(0.1, 0.5)),
            suggestions=kwargs.get("suggestions", [fake.sentence() for _ in range(random.randint(0, 3))]),
            processing_time_ms=kwargs.get("processing_time_ms", random.randint(50, 500)),
            validated_at=kwargs.get("validated_at", datetime.now(timezone.utc))
        )

    @staticmethod
    def create_duplicate_check_result(
        is_duplicate: bool = False,
        **kwargs
    ) -> DuplicateCheckResult:
        """Create a test DuplicateCheckResult object."""
        return DuplicateCheckResult(
            is_duplicate=is_duplicate,
            similarity_score=kwargs.get("similarity_score", 
                random.uniform(0.8, 1.0) if is_duplicate else random.uniform(0.0, 0.3)
            ),
            similar_ideas=kwargs.get("similar_ideas", [
                TestDataFactory.create_idea().dict() 
                for _ in range(random.randint(1, 3))
            ] if is_duplicate else []),
            check_performed_at=kwargs.get("check_performed_at", datetime.now(timezone.utc))
        )

    @staticmethod
    def create_startup_ideas_batch(count: int = 5) -> List[IdeaDraft]:
        """Create a batch of diverse startup ideas for testing."""
        idea_templates = [
            {
                "title": "AI-Powered Task Manager",
                "description": "Intelligent task management system that learns from user behavior to optimize productivity and prioritize work automatically.",
                "category": IdeaCategory.PRODUCTIVITY
            },
            {
                "title": "Sustainable Fashion Marketplace",
                "description": "Online platform connecting eco-conscious consumers with sustainable fashion brands and second-hand clothing.",
                "category": IdeaCategory.ECOMMERCE
            },
            {
                "title": "EdTech VR Platform",
                "description": "Virtual reality educational platform for immersive learning experiences in science, history, and languages.",
                "category": IdeaCategory.EDUCATION
            },
            {
                "title": "HealthTech Monitoring App",
                "description": "Mobile app for continuous health monitoring using wearable devices and AI-powered health insights.",
                "category": IdeaCategory.HEALTH
            },
            {
                "title": "FinTech Micro-Investment",
                "description": "Automated micro-investment platform that rounds up purchases and invests spare change in diversified portfolios.",
                "category": IdeaCategory.FINTECH
            }
        ]
        
        ideas = []
        for i in range(min(count, len(idea_templates))):
            template = idea_templates[i]
            ideas.append(TestDataFactory.create_idea_draft(
                title=template["title"],
                description=template["description"],
                category=template["category"]
            ))
        
        # Fill remaining with random ideas if needed
        for _ in range(count - len(ideas)):
            ideas.append(TestDataFactory.create_idea_draft())
        
        return ideas

    @staticmethod
    def create_test_config() -> Dict[str, Any]:
        """Create test configuration settings."""
        return {
            "database": {
                "url": "postgresql://test:test@localhost:5432/test_db",
                "pool_size": 5,
                "timeout": 30
            },
            "validation": {
                "quality_threshold": 0.7,
                "similarity_threshold": 0.8,
                "max_retries": 3
            },
            "processing": {
                "batch_size": 10,
                "timeout_seconds": 300,
                "max_concurrent_tasks": 5
            },
            "ai_services": {
                "openai_model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.1
            }
        }

    @staticmethod
    def create_performance_test_data(size: str = "small") -> Dict[str, Any]:
        """Create test data sets for performance testing."""
        sizes = {
            "small": 10,
            "medium": 100,
            "large": 1000,
            "xlarge": 10000
        }
        
        count = sizes.get(size, 10)
        
        return {
            "ideas": [TestDataFactory.create_idea_draft().dict() for _ in range(count)],
            "expected_processing_time": {
                "small": 5.0,    # seconds
                "medium": 30.0,
                "large": 300.0,
                "xlarge": 1800.0
            }.get(size, 5.0),
            "memory_limit_mb": {
                "small": 100,
                "medium": 500,
                "large": 2000,
                "xlarge": 8000
            }.get(size, 100)
        }


class MockDataProvider:
    """Provider for mock data and responses."""

    @staticmethod
    def mock_api_response(success: bool = True) -> Dict[str, Any]:
        """Create a mock API response."""
        if success:
            return {
                "status": "success",
                "data": {
                    "id": str(uuid4()),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "result": TestDataFactory.create_validation_result().dict()
                },
                "message": "Processing completed successfully"
            }
        else:
            return {
                "status": "error",
                "error": {
                    "code": random.choice(["VALIDATION_FAILED", "TIMEOUT", "INVALID_INPUT"]),
                    "message": fake.sentence(),
                    "details": {"field": fake.word(), "reason": fake.sentence()}
                },
                "message": "Processing failed"
            }

    @staticmethod
    def mock_database_records(count: int = 5) -> List[Dict[str, Any]]:
        """Create mock database records."""
        return [
            {
                "id": str(uuid4()),
                "title": fake.catch_phrase(),
                "description": fake.text(max_nb_chars=300),
                "created_at": fake.date_time_this_year().isoformat(),
                "status": random.choice(["draft", "processing", "completed", "failed"]),
                "metadata": {"test": True, "source": "mock"}
            }
            for _ in range(count)
        ]

    @staticmethod
    def mock_ai_service_response(model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Create a mock AI service response."""
        return {
            "id": f"chatcmpl-{fake.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": fake.text(max_nb_chars=500)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": random.randint(50, 200),
                "completion_tokens": random.randint(100, 400),
                "total_tokens": random.randint(150, 600)
            }
        }