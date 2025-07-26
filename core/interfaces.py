# core/interfaces.py
"""
Core service interfaces for dependency injection and loose coupling.
Defines contracts between services to enable testability and modularity.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID


class IAlertManager(ABC):
    """Interface for alert management services"""
    
    @abstractmethod
    def record_alert(self, message: str, severity: str = "warning") -> None:
        """Record an alert with specified severity"""
        pass
        
    @abstractmethod
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded alerts"""
        pass


class IBudgetSentinel(ABC):
    """Interface for budget monitoring services"""
    
    @abstractmethod
    def check_usage(self, current_usage: float, context: str = "") -> bool:
        """Check if usage is within budget limits"""
        pass
        
    @abstractmethod
    def is_budget_exceeded(self) -> bool:
        """Check if budget has been exceeded"""
        pass


class IEvidenceCollector(ABC):
    """Interface for evidence collection services"""
    
    @abstractmethod
    def collect_evidence(self, claim: str, num_sources: int = 3) -> List[str]:
        """Collect evidence sources for a given claim"""
        pass
        
    @abstractmethod
    def verify_sources(self, sources: List[str]) -> Dict[str, bool]:
        """Verify the validity of evidence sources"""
        pass


class ISearchTool(ABC):
    """Interface for search tool implementations"""
    
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[str]:
        """Perform search and return list of URLs"""
        pass


class IIdeaRepository(ABC):
    """Interface for idea storage and retrieval"""
    
    @abstractmethod
    def create_idea(self, idea_data: Dict[str, Any]) -> UUID:
        """Create a new idea and return its ID"""
        pass
        
    @abstractmethod
    def get_idea(self, idea_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve an idea by ID"""
        pass
        
    @abstractmethod
    def update_idea(self, idea_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update an existing idea"""
        pass
        
    @abstractmethod
    def delete_idea(self, idea_id: UUID) -> bool:
        """Delete an idea"""
        pass
        
    @abstractmethod
    def list_ideas(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """List ideas with pagination"""
        pass


class IInvestorScorer(ABC):
    """Interface for investor scoring services"""
    
    @abstractmethod
    def score_pitch(self, pitch_data: Dict[str, Any], investor_profile: str) -> Dict[str, Any]:
        """Score a pitch against investor criteria"""
        pass
        
    @abstractmethod
    def load_investor_profiles(self) -> List[str]:
        """Load available investor profiles"""
        pass


class IDeckGenerator(ABC):
    """Interface for pitch deck generation services"""
    
    @abstractmethod
    def generate_deck(self, idea_data: Dict[str, Any]) -> str:
        """Generate a pitch deck and return file path"""
        pass
        
    @abstractmethod
    def validate_deck_template(self, template_path: str) -> bool:
        """Validate deck template exists and is usable"""
        pass


class IServiceRegistry(ABC):
    """Interface for service discovery and registration"""
    
    @abstractmethod
    def register(self, service_name: str, service_instance: Any) -> None:
        """Register a service instance"""
        pass
        
    @abstractmethod
    def get(self, service_name: str) -> Any:
        """Retrieve a registered service"""
        pass
        
    @abstractmethod
    def list_services(self) -> List[str]:
        """List all registered service names"""
        pass


class IConfigurationProvider(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        pass
        
    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
        
    @abstractmethod
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from file"""
        pass


# Type aliases for common callback functions
AlertCallback = Callable[[str], None]
HaltCallback = Callable[[str, str], None]
SearchCallback = Callable[[str, int], List[str]]
ValidationCallback = Callable[[Any], bool]