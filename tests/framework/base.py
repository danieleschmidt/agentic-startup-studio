"""
Base classes for validation framework.

This module provides the base validator interface and result classes
for the validation framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import ValidationArea


class ValidationResult:
    """
    Container for validation results.
    
    This class holds the result of a validation operation including
    success status, messages, and detailed information.
    """
    
    def __init__(
        self,
        validation_id: str,
        area: ValidationArea,
        status: str,  # ValidationStatus enum value
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a ValidationResult."""
        self.validation_id = validation_id
        self.area = area
        self.status = status
        self.message = message
        self.details = details or {}
        self.errors = []
        self.metrics = {}
    
    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        # Change status to failed when error is added
        from .config import ValidationStatus
        self.status = ValidationStatus.FAILED
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the result."""
        self.metrics[name] = value
    
    def is_success(self) -> bool:
        """Check if validation was successful."""
        from .config import ValidationStatus
        return self.status == ValidationStatus.PASSED
    
    
    def __str__(self) -> str:
        """String representation of the validation result."""
        return f"{self.validation_id} ({self.area.value}): {self.status} - {self.message}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the validation result."""
        return (
            f"ValidationResult(validation_id='{self.validation_id}', "
            f"area={self.area}, status={self.status}, "
            f"message='{self.message}', details={self.details})"
        )


class ValidationContext:
    """
    Context for validation execution.
    
    This class provides test data and configuration context
    for validation operations.
    """
    
    def __init__(
        self,
        test_data: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        dependencies: Optional[list] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3
    ):
        """Initialize a ValidationContext."""
        self.test_data = test_data or {}
        self.configuration = configuration or {}
        self.environment = environment or {}
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.max_retries = max_retries
    
    def __str__(self) -> str:
        """String representation of the validation context."""
        return f"ValidationContext(test_data={len(self.test_data)} items, timeout={self.timeout})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the validation context."""
        return (
            f"ValidationContext(test_data={self.test_data}, "
            f"configuration={self.configuration}, timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    
    This class defines the interface that all validators must implement
    to be compatible with the validation engine.
    """
    
    def __init__(self, config=None):
        """Initialize the validator with optional configuration."""
        self.config = config
    
    @property
    @abstractmethod
    def area(self) -> ValidationArea:
        """Return the validation area this validator covers."""
        pass
    
    @abstractmethod
    def validate(self, data: Any = None, **kwargs) -> ValidationResult:
        """
        Perform validation on the provided data.
        
        Args:
            data: The data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult: The result of the validation
        """
        pass
    
    async def validate_async(self, data: Any = None, **kwargs) -> ValidationResult:
        """
        Async validation method (optional override).
        
        Default implementation calls the sync validate method.
        Validators can override this for true async validation.
        
        Args:
            data: The data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult: The result of the validation
        """
        return self.validate(data, **kwargs)
    
    def __str__(self) -> str:
        """String representation of the validator."""
        return f"{self.__class__.__name__}({self.area.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the validator."""
        return f"{self.__class__.__name__}(area={self.area}, config={self.config})"