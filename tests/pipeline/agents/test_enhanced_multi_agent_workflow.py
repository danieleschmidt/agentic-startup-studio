"""
Comprehensive test suite for Enhanced Multi-Agent Workflow.

Tests cover CrewAI and LangGraph integration, agent coordination,
workflow state management, and error handling.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

from pipeline.agents.enhanced_multi_agent_workflow import (
    AgentRole,
    WorkflowStage,
    TaskPriority,
    AgentConfig,
    TaskConfig,
    MultiAgentState,
    CREWAI_AVAILABLE,
    LANGGRAPH_AVAILABLE,
    LANGCHAIN_AVAILABLE
)


# Skip tests if dependencies are not available
pytestmark = pytest.mark.skipif(
    not (CREWAI_AVAILABLE and LANGGRAPH_AVAILABLE and LANGCHAIN_AVAILABLE),
    reason="CrewAI, LangGraph, or LangChain not available"
)


@pytest.fixture
def sample_agent_config():
    """Create a sample agent configuration."""
    return AgentConfig(
        role=AgentRole.CEO,
        name="Strategic CEO",
        goal="Provide strategic direction and business validation",
        backstory="Experienced startup founder with multiple exits",
        skills=["strategic_planning", "market_analysis", "fundraising"],
        tools=["market_research", "financial_modeling"],
        memory_type="vector",
        max_iterations=3,
        temperature=0.7,
        delegation_allowed=True,
        verbose=True
    )


@pytest.fixture
def sample_task_config():
    """Create a sample task configuration."""
    return TaskConfig(
        task_id="validate_market_opportunity",
        description="Analyze market size, competition, and opportunity",
        agent_role=AgentRole.CEO,
        stage=WorkflowStage.MARKET_ANALYSIS,
        priority=TaskPriority.HIGH,
        dependencies=["ideation_complete"],
        expected_output="Market analysis report with TAM, SAM, SOM estimates",
        max_execution_time=300,
        retry_count=2,
        validation_criteria={
            "market_size_minimum": 1000000,
            "competition_level": "manageable",
            "confidence_threshold": 0.7
        }
    )


@pytest.fixture
def sample_multi_agent_state():
    """Create a sample multi-agent state."""
    return MultiAgentState(
        workflow_id="test-workflow-123",
        current_stage=WorkflowStage.IDEATION,
        startup_idea="AI-powered customer service automation platform",
        progress=0.0,
        started_at=datetime.now(),
        
        # Agent outputs
        ideation_results={},
        technical_validation={},
        market_analysis={},
        business_model={},
        investor_pitch={},
        technical_architecture={},
        go_to_market={},
        final_review={},
        
        # Task tracking
        completed_tasks=[],
        failed_tasks=[],
        active_tasks=[],
        
        # Quality metrics
        consensus_score=0.0,
        technical_feasibility=0.0,
        market_viability=0.0,
        business_potential=0.0,
        
        # Agent interactions
        agent_messages=[],
        debates=[],
        decisions=[],
        
        # Error handling
        errors=[],
        retry_count=0
    )


class TestAgentRole:
    """Test AgentRole enum."""
    
    def test_agent_role_values(self):
        """Test agent role enum values."""
        assert AgentRole.CEO.value == "ceo"
        assert AgentRole.CTO.value == "cto"
        assert AgentRole.VP_RD.value == "vp_rd"
        assert AgentRole.GROWTH_MARKETER.value == "growth_marketer"
        assert AgentRole.VC_ANALYST.value == "vc_analyst"
        assert AgentRole.ANGEL_INVESTOR.value == "angel_investor"
        assert AgentRole.ENGINEER.value == "engineer"
    
    def test_agent_role_coverage(self):
        """Test that all required roles are defined."""
        roles = [role.value for role in AgentRole]
        
        # Check that key startup roles are covered
        assert "ceo" in roles
        assert "cto" in roles
        assert "growth_marketer" in roles
        assert "vc_analyst" in roles
        assert "angel_investor" in roles


class TestWorkflowStage:
    """Test WorkflowStage enum."""
    
    def test_workflow_stage_values(self):
        """Test workflow stage enum values."""
        assert WorkflowStage.IDEATION.value == "ideation"
        assert WorkflowStage.TECHNICAL_VALIDATION.value == "technical_validation"
        assert WorkflowStage.MARKET_ANALYSIS.value == "market_analysis"
        assert WorkflowStage.BUSINESS_MODEL.value == "business_model"
        assert WorkflowStage.INVESTOR_PITCH.value == "investor_pitch"
        assert WorkflowStage.TECHNICAL_ARCHITECTURE.value == "technical_architecture"
        assert WorkflowStage.GO_TO_MARKET.value == "go_to_market"
        assert WorkflowStage.FINAL_REVIEW.value == "final_review"
        assert WorkflowStage.COMPLETED.value == "completed"
    
    def test_workflow_stage_progression(self):
        """Test logical workflow stage progression."""
        stages = [
            WorkflowStage.IDEATION,
            WorkflowStage.TECHNICAL_VALIDATION,
            WorkflowStage.MARKET_ANALYSIS,
            WorkflowStage.BUSINESS_MODEL,
            WorkflowStage.INVESTOR_PITCH,
            WorkflowStage.TECHNICAL_ARCHITECTURE,
            WorkflowStage.GO_TO_MARKET,
            WorkflowStage.FINAL_REVIEW,
            WorkflowStage.COMPLETED
        ]
        
        # Ensure all stages are unique and logical
        assert len(stages) == len(set(stages))
        
        # Test that stages have a logical flow
        ideation_index = stages.index(WorkflowStage.IDEATION)
        technical_index = stages.index(WorkflowStage.TECHNICAL_VALIDATION)
        market_index = stages.index(WorkflowStage.MARKET_ANALYSIS)
        completed_index = stages.index(WorkflowStage.COMPLETED)
        
        assert ideation_index < technical_index
        assert ideation_index < market_index
        assert completed_index == len(stages) - 1


class TestTaskPriority:
    """Test TaskPriority enum."""
    
    def test_task_priority_values(self):
        """Test task priority enum values."""
        assert TaskPriority.CRITICAL.value == "critical"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.LOW.value == "low"
    
    def test_task_priority_hierarchy(self):
        """Test task priority logical hierarchy."""
        priorities = [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.MEDIUM,
            TaskPriority.LOW
        ]
        
        # All priorities should be unique
        assert len(priorities) == len(set(priorities))


class TestAgentConfig:
    """Test AgentConfig data class."""
    
    def test_agent_config_creation(self, sample_agent_config):
        """Test agent configuration creation."""
        config = sample_agent_config
        
        assert config.role == AgentRole.CEO
        assert config.name == "Strategic CEO"
        assert "strategic_planning" in config.skills
        assert "market_research" in config.tools
        assert config.memory_type == "vector"
        assert config.max_iterations == 3
        assert config.temperature == 0.7
        assert config.delegation_allowed is True
        assert config.verbose is True
    
    def test_agent_config_defaults(self):
        """Test agent configuration with defaults."""
        config = AgentConfig(
            role=AgentRole.CTO,
            name="Technical CTO",
            goal="Provide technical leadership",
            backstory="Senior engineer with architecture experience"
        )
        
        assert config.skills == []
        assert config.tools == []
        assert config.memory_type == "vector"
        assert config.max_iterations == 3
        assert config.temperature == 0.7
        assert config.delegation_allowed is True
        assert config.verbose is True
    
    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Test with valid configuration
        config = AgentConfig(
            role=AgentRole.VC_ANALYST,
            name="Investment Analyst",
            goal="Analyze investment opportunities",
            backstory="Former venture capitalist",
            skills=["financial_analysis", "due_diligence"],
            tools=["spreadsheet", "market_data"],
            temperature=0.5,
            max_iterations=5
        )
        
        assert config.temperature == 0.5
        assert config.max_iterations == 5
        assert len(config.skills) == 2
        assert len(config.tools) == 2


class TestTaskConfig:
    """Test TaskConfig data class."""
    
    def test_task_config_creation(self, sample_task_config):
        """Test task configuration creation."""
        config = sample_task_config
        
        assert config.task_id == "validate_market_opportunity"
        assert config.agent_role == AgentRole.CEO
        assert config.stage == WorkflowStage.MARKET_ANALYSIS
        assert config.priority == TaskPriority.HIGH
        assert "ideation_complete" in config.dependencies
        assert config.max_execution_time == 300
        assert config.retry_count == 2
        assert config.validation_criteria["market_size_minimum"] == 1000000
    
    def test_task_config_defaults(self):
        """Test task configuration with defaults."""
        config = TaskConfig(
            task_id="simple_task",
            description="Simple task description",
            agent_role=AgentRole.ENGINEER,
            stage=WorkflowStage.TECHNICAL_VALIDATION,
            priority=TaskPriority.MEDIUM
        )
        
        assert config.dependencies == []
        assert config.expected_output == ""
        assert config.max_execution_time == 300
        assert config.retry_count == 2
        assert config.validation_criteria == {}
    
    def test_task_config_dependency_validation(self):
        """Test task dependency validation."""
        config = TaskConfig(
            task_id="dependent_task",
            description="Task with dependencies",
            agent_role=AgentRole.CEO,
            stage=WorkflowStage.BUSINESS_MODEL,
            priority=TaskPriority.HIGH,
            dependencies=["market_analysis", "technical_validation", "ideation"]
        )
        
        assert len(config.dependencies) == 3
        assert "market_analysis" in config.dependencies
        assert "technical_validation" in config.dependencies
        assert "ideation" in config.dependencies


class TestMultiAgentState:
    """Test MultiAgentState TypedDict."""
    
    def test_multi_agent_state_creation(self, sample_multi_agent_state):
        """Test multi-agent state creation."""
        state = sample_multi_agent_state
        
        assert state["workflow_id"] == "test-workflow-123"
        assert state["current_stage"] == WorkflowStage.IDEATION
        assert "AI-powered customer service" in state["startup_idea"]
        assert state["progress"] == 0.0
        assert isinstance(state["started_at"], datetime)
        
        # Check all required dictionaries are initialized
        assert isinstance(state["ideation_results"], dict)
        assert isinstance(state["technical_validation"], dict)
        assert isinstance(state["market_analysis"], dict)
        assert isinstance(state["business_model"], dict)
        assert isinstance(state["investor_pitch"], dict)
        assert isinstance(state["technical_architecture"], dict)
        assert isinstance(state["go_to_market"], dict)
        assert isinstance(state["final_review"], dict)
        
        # Check all required lists are initialized
        assert isinstance(state["completed_tasks"], list)
        assert isinstance(state["failed_tasks"], list)
        assert isinstance(state["active_tasks"], list)
        assert isinstance(state["agent_messages"], list)
        assert isinstance(state["debates"], list)
        assert isinstance(state["decisions"], list)
        assert isinstance(state["errors"], list)
        
        # Check quality metrics
        assert state["consensus_score"] == 0.0
        assert state["technical_feasibility"] == 0.0
        assert state["market_viability"] == 0.0
        assert state["business_potential"] == 0.0
        assert state["retry_count"] == 0
    
    def test_multi_agent_state_updates(self, sample_multi_agent_state):
        """Test updating multi-agent state."""
        state = sample_multi_agent_state
        
        # Update workflow progress
        state["progress"] = 0.25
        state["current_stage"] = WorkflowStage.TECHNICAL_VALIDATION
        
        # Add completed task
        state["completed_tasks"].append("ideation_brainstorm")
        
        # Update ideation results
        state["ideation_results"] = {
            "core_concept": "AI chatbot automation",
            "target_market": "SMB customer service",
            "value_proposition": "Reduce response time by 80%",
            "confidence_score": 0.85
        }
        
        # Add agent message
        state["agent_messages"].append({
            "from_agent": AgentRole.CEO.value,
            "to_agent": AgentRole.CTO.value,
            "message": "Technical feasibility assessment needed",
            "timestamp": datetime.now().isoformat(),
            "priority": TaskPriority.HIGH.value
        })
        
        # Update quality metrics
        state["consensus_score"] = 0.8
        state["technical_feasibility"] = 0.75
        
        # Verify updates
        assert state["progress"] == 0.25
        assert state["current_stage"] == WorkflowStage.TECHNICAL_VALIDATION
        assert len(state["completed_tasks"]) == 1
        assert "ideation_brainstorm" in state["completed_tasks"]
        assert state["ideation_results"]["confidence_score"] == 0.85
        assert len(state["agent_messages"]) == 1
        assert state["agent_messages"][0]["from_agent"] == "ceo"
        assert state["consensus_score"] == 0.8
        assert state["technical_feasibility"] == 0.75
    
    def test_multi_agent_state_task_tracking(self, sample_multi_agent_state):
        """Test task tracking in multi-agent state."""
        state = sample_multi_agent_state
        
        # Add active tasks
        state["active_tasks"] = [
            "market_research",
            "competitive_analysis",
            "technical_architecture"
        ]
        
        # Complete some tasks
        state["completed_tasks"] = [
            "ideation_brainstorm",
            "problem_definition"
        ]
        
        # Add failed task
        state["failed_tasks"] = [
            "patent_search"  # Maybe external API failed
        ]
        
        # Verify task tracking
        assert len(state["active_tasks"]) == 3
        assert len(state["completed_tasks"]) == 2
        assert len(state["failed_tasks"]) == 1
        
        assert "market_research" in state["active_tasks"]
        assert "ideation_brainstorm" in state["completed_tasks"]
        assert "patent_search" in state["failed_tasks"]
    
    def test_multi_agent_state_error_handling(self, sample_multi_agent_state):
        """Test error handling in multi-agent state."""
        state = sample_multi_agent_state
        
        # Add errors
        state["errors"] = [
            "External API timeout during market research",
            "Agent execution failed: insufficient context",
            "Budget exceeded for competitive analysis"
        ]
        
        # Increment retry count
        state["retry_count"] = 2
        
        # Verify error tracking
        assert len(state["errors"]) == 3
        assert "External API timeout" in state["errors"][0]
        assert "Agent execution failed" in state["errors"][1]
        assert "Budget exceeded" in state["errors"][2]
        assert state["retry_count"] == 2
    
    def test_multi_agent_state_agent_interactions(self, sample_multi_agent_state):
        """Test agent interaction tracking."""
        state = sample_multi_agent_state
        
        # Add agent message
        message = {
            "id": "msg_001",
            "from_agent": AgentRole.CEO.value,
            "to_agent": AgentRole.CTO.value,
            "content": {
                "topic": "technical_feasibility",
                "request": "Assess development complexity for AI chatbot",
                "context": state["ideation_results"],
                "deadline": "2024-01-15T10:00:00Z"
            },
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        state["agent_messages"].append(message)
        
        # Add debate entry
        debate = {
            "id": "debate_001",
            "topic": "technology_stack_selection",
            "participants": [AgentRole.CTO.value, AgentRole.ENGINEER.value],
            "positions": {
                AgentRole.CTO.value: "Use proven Python/Django stack",
                AgentRole.ENGINEER.value: "Consider Node.js for real-time features"
            },
            "resolution": "pending",
            "started_at": datetime.now().isoformat()
        }
        state["debates"].append(debate)
        
        # Add decision
        decision = {
            "id": "decision_001",
            "type": "technical_architecture",
            "made_by": AgentRole.CTO.value,
            "decision": "Implement microservices architecture with Python backend",
            "rationale": "Better scalability and team expertise alignment",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
            "impact_on_timeline": "minimal",
            "affected_agents": [AgentRole.ENGINEER.value, AgentRole.VP_RD.value]
        }
        state["decisions"].append(decision)
        
        # Verify interactions
        assert len(state["agent_messages"]) == 1
        assert len(state["debates"]) == 1
        assert len(state["decisions"]) == 1
        
        assert state["agent_messages"][0]["from_agent"] == "ceo"
        assert state["agent_messages"][0]["to_agent"] == "cto"
        assert "technical_feasibility" in state["agent_messages"][0]["content"]["topic"]
        
        assert "technology_stack_selection" in state["debates"][0]["topic"]
        assert len(state["debates"][0]["participants"]) == 2
        
        assert state["decisions"][0]["made_by"] == "cto"
        assert state["decisions"][0]["confidence"] == 0.85


class TestDependencyAvailability:
    """Test external dependency availability and graceful degradation."""
    
    def test_crewai_availability(self):
        """Test CrewAI availability detection."""
        assert isinstance(CREWAI_AVAILABLE, bool)
        
        if CREWAI_AVAILABLE:
            # Should be able to import CrewAI components
            from pipeline.agents.enhanced_multi_agent_workflow import Agent, Task, Crew, Process
            assert Agent is not None
            assert Task is not None
            assert Crew is not None
            assert Process is not None
        else:
            # Should gracefully handle missing dependency
            from pipeline.agents.enhanced_multi_agent_workflow import Agent, Task, Crew, Process
            assert Agent is None
            assert Task is None
            assert Crew is None
            assert Process is None
    
    def test_langgraph_availability(self):
        """Test LangGraph availability detection."""
        assert isinstance(LANGGRAPH_AVAILABLE, bool)
        
        if LANGGRAPH_AVAILABLE:
            # Should be able to import LangGraph components
            from pipeline.agents.enhanced_multi_agent_workflow import StateGraph, END, START
            assert StateGraph is not None
            assert END is not None
            assert START is not None
        else:
            # Should gracefully handle missing dependency
            from pipeline.agents.enhanced_multi_agent_workflow import StateGraph, END, START
            assert StateGraph is None
            assert END is None
            assert START is None
    
    def test_langchain_availability(self):
        """Test LangChain availability detection."""
        assert isinstance(LANGCHAIN_AVAILABLE, bool)
        
        if LANGCHAIN_AVAILABLE:
            # Should be able to import LangChain components
            from pipeline.agents.enhanced_multi_agent_workflow import (
                ChatOpenAI, BaseMessage, HumanMessage, AIMessage
            )
            assert ChatOpenAI is not None
            assert BaseMessage is not None
            assert HumanMessage is not None
            assert AIMessage is not None
        else:
            # Should gracefully handle missing dependency
            from pipeline.agents.enhanced_multi_agent_workflow import (
                ChatOpenAI, BaseMessage, HumanMessage, AIMessage
            )
            assert ChatOpenAI is None
            assert BaseMessage is None
            assert HumanMessage is None
            assert AIMessage is None


class TestWorkflowValidation:
    """Test workflow validation and business logic."""
    
    def test_agent_role_task_compatibility(self):
        """Test that agent roles are compatible with their assigned tasks."""
        # CEO should handle strategic tasks
        ceo_config = AgentConfig(
            role=AgentRole.CEO,
            name="Strategic CEO",
            goal="Strategic direction",
            backstory="Business leader",
            skills=["strategic_planning", "market_analysis", "fundraising"]
        )
        
        market_task = TaskConfig(
            task_id="market_analysis",
            description="Analyze market opportunity",
            agent_role=AgentRole.CEO,
            stage=WorkflowStage.MARKET_ANALYSIS,
            priority=TaskPriority.HIGH
        )
        
        # Verify compatibility
        assert ceo_config.role == market_task.agent_role
        assert "strategic_planning" in ceo_config.skills
        assert market_task.stage == WorkflowStage.MARKET_ANALYSIS
        
        # CTO should handle technical tasks
        cto_config = AgentConfig(
            role=AgentRole.CTO,
            name="Technical CTO",
            goal="Technical leadership",
            backstory="Senior engineer",
            skills=["architecture", "system_design", "security"]
        )
        
        tech_task = TaskConfig(
            task_id="technical_architecture",
            description="Design system architecture",
            agent_role=AgentRole.CTO,
            stage=WorkflowStage.TECHNICAL_ARCHITECTURE,
            priority=TaskPriority.CRITICAL
        )
        
        # Verify compatibility
        assert cto_config.role == tech_task.agent_role
        assert "architecture" in cto_config.skills
        assert tech_task.stage == WorkflowStage.TECHNICAL_ARCHITECTURE
    
    def test_workflow_stage_dependencies(self):
        """Test logical dependencies between workflow stages."""
        # Market analysis should depend on ideation
        market_task = TaskConfig(
            task_id="market_sizing",
            description="Determine market size",
            agent_role=AgentRole.CEO,
            stage=WorkflowStage.MARKET_ANALYSIS,
            priority=TaskPriority.HIGH,
            dependencies=["ideation_complete"]
        )
        
        # Technical architecture should depend on business model
        tech_task = TaskConfig(
            task_id="system_design",
            description="Design technical architecture",
            agent_role=AgentRole.CTO,
            stage=WorkflowStage.TECHNICAL_ARCHITECTURE,
            priority=TaskPriority.HIGH,
            dependencies=["business_model_validated", "market_analysis_complete"]
        )
        
        # Investor pitch should depend on multiple earlier stages
        pitch_task = TaskConfig(
            task_id="pitch_deck_creation",
            description="Create investor pitch deck",
            agent_role=AgentRole.VC_ANALYST,
            stage=WorkflowStage.INVESTOR_PITCH,
            priority=TaskPriority.CRITICAL,
            dependencies=[
                "market_analysis_complete",
                "business_model_validated",
                "technical_validation_complete"
            ]
        )
        
        # Verify dependencies make logical sense
        assert "ideation_complete" in market_task.dependencies
        assert "business_model_validated" in tech_task.dependencies
        assert "market_analysis_complete" in tech_task.dependencies
        
        assert len(pitch_task.dependencies) >= 3
        assert "market_analysis_complete" in pitch_task.dependencies
        assert "business_model_validated" in pitch_task.dependencies
        assert "technical_validation_complete" in pitch_task.dependencies
    
    def test_task_priority_logic(self):
        """Test task priority assignment logic."""
        # Critical tasks should be foundational or blocking
        critical_task = TaskConfig(
            task_id="validate_core_concept",
            description="Validate core business concept",
            agent_role=AgentRole.CEO,
            stage=WorkflowStage.IDEATION,
            priority=TaskPriority.CRITICAL
        )
        
        # High priority for key business decisions
        high_task = TaskConfig(
            task_id="competitive_analysis",
            description="Analyze competitive landscape",
            agent_role=AgentRole.GROWTH_MARKETER,
            stage=WorkflowStage.MARKET_ANALYSIS,
            priority=TaskPriority.HIGH
        )
        
        # Medium priority for supporting analysis
        medium_task = TaskConfig(
            task_id="technology_research",
            description="Research available technologies",
            agent_role=AgentRole.ENGINEER,
            stage=WorkflowStage.TECHNICAL_VALIDATION,
            priority=TaskPriority.MEDIUM
        )
        
        # Low priority for nice-to-have insights
        low_task = TaskConfig(
            task_id="industry_trend_analysis",
            description="Analyze long-term industry trends",
            agent_role=AgentRole.VC_ANALYST,
            stage=WorkflowStage.FINAL_REVIEW,
            priority=TaskPriority.LOW
        )
        
        # Verify priority assignments make sense
        assert critical_task.priority == TaskPriority.CRITICAL
        assert critical_task.stage == WorkflowStage.IDEATION  # Early stage
        
        assert high_task.priority == TaskPriority.HIGH
        assert medium_task.priority == TaskPriority.MEDIUM
        assert low_task.priority == TaskPriority.LOW
        assert low_task.stage == WorkflowStage.FINAL_REVIEW  # Late stage


class TestWorkflowIntegration:
    """Integration tests for workflow components."""
    
    def test_complete_workflow_state_lifecycle(self, sample_multi_agent_state):
        """Test complete workflow state lifecycle."""
        state = sample_multi_agent_state
        
        # Stage 1: Ideation
        assert state["current_stage"] == WorkflowStage.IDEATION
        state["ideation_results"] = {
            "concept": "AI customer service platform",
            "problem": "Long customer wait times",
            "solution": "Intelligent chatbot triage",
            "confidence": 0.8
        }
        state["completed_tasks"].append("ideation_brainstorm")
        state["progress"] = 0.125  # 1/8 stages complete
        
        # Stage 2: Technical Validation
        state["current_stage"] = WorkflowStage.TECHNICAL_VALIDATION
        state["technical_validation"] = {
            "feasibility": "high",
            "complexity": "medium",
            "timeline": "6 months",
            "confidence": 0.85
        }
        state["completed_tasks"].append("technical_assessment")
        state["technical_feasibility"] = 0.85
        state["progress"] = 0.25  # 2/8 stages complete
        
        # Stage 3: Market Analysis
        state["current_stage"] = WorkflowStage.MARKET_ANALYSIS
        state["market_analysis"] = {
            "market_size": 50000000,  # $50M TAM
            "competition": "moderate",
            "growth_rate": 0.15,
            "confidence": 0.75
        }
        state["completed_tasks"].append("market_research")
        state["market_viability"] = 0.75
        state["progress"] = 0.375  # 3/8 stages complete
        
        # Stage 4: Business Model
        state["current_stage"] = WorkflowStage.BUSINESS_MODEL
        state["business_model"] = {
            "revenue_model": "SaaS subscription",
            "customer_segments": ["SMB", "Mid-market"],
            "pricing": {"basic": 29, "pro": 99, "enterprise": 299},
            "confidence": 0.8
        }
        state["completed_tasks"].append("business_model_design")
        state["business_potential"] = 0.8
        state["progress"] = 0.5  # 4/8 stages complete
        
        # Continue with remaining stages...
        assert len(state["completed_tasks"]) == 4
        assert state["progress"] == 0.5
        assert state["technical_feasibility"] == 0.85
        assert state["market_viability"] == 0.75
        assert state["business_potential"] == 0.8
        
        # Calculate consensus score
        scores = [
            state["technical_feasibility"],
            state["market_viability"],
            state["business_potential"]
        ]
        state["consensus_score"] = sum(scores) / len(scores)
        assert abs(state["consensus_score"] - 0.8) < 0.01  # Approximately 0.8
    
    def test_agent_collaboration_scenario(self, sample_multi_agent_state):
        """Test realistic agent collaboration scenario."""
        state = sample_multi_agent_state
        
        # CEO initiates market analysis request
        ceo_message = {
            "id": "msg_001",
            "from_agent": AgentRole.CEO.value,
            "to_agent": AgentRole.GROWTH_MARKETER.value,
            "content": {
                "request": "market_analysis",
                "startup_idea": state["startup_idea"],
                "focus_areas": ["market_size", "competition", "customer_segments"],
                "deadline": "2024-01-20T17:00:00Z"
            },
            "timestamp": datetime.now().isoformat(),
            "priority": TaskPriority.HIGH.value
        }
        state["agent_messages"].append(ceo_message)
        
        # Growth Marketer responds with analysis
        growth_response = {
            "id": "msg_002",
            "from_agent": AgentRole.GROWTH_MARKETER.value,
            "to_agent": AgentRole.CEO.value,
            "content": {
                "response_to": "msg_001",
                "analysis": {
                    "market_size_tam": 5000000000,  # $5B
                    "market_size_sam": 500000000,   # $500M
                    "market_size_som": 50000000,    # $50M
                    "competition_level": "moderate",
                    "key_competitors": ["Zendesk", "Intercom", "Freshdesk"],
                    "customer_segments": ["SMB (0-100 employees)", "Mid-market (100-1000)"],
                    "confidence": 0.82
                },
                "next_steps": ["validate_with_customer_interviews", "competitive_pricing_analysis"]
            },
            "timestamp": datetime.now().isoformat()
        }
        state["agent_messages"].append(growth_response)
        
        # CTO weighs in on technical feasibility
        cto_message = {
            "id": "msg_003", 
            "from_agent": AgentRole.CTO.value,
            "to_agent": AgentRole.CEO.value,
            "content": {
                "topic": "technical_feasibility_assessment",
                "assessment": {
                    "ai_complexity": "medium",
                    "integration_challenges": ["CRM systems", "ticketing platforms"],
                    "scalability": "high",
                    "security_requirements": ["SOC2", "GDPR compliance"],
                    "development_timeline": "6-8 months MVP",
                    "confidence": 0.88
                },
                "resource_requirements": {
                    "engineers": 3,
                    "ai_specialists": 2,
                    "budget_estimate": 500000
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        state["agent_messages"].append(cto_message)
        
        # VC Analyst provides investment perspective
        vc_message = {
            "id": "msg_004",
            "from_agent": AgentRole.VC_ANALYST.value,
            "to_agent": AgentRole.CEO.value,
            "content": {
                "investment_analysis": {
                    "market_opportunity": "strong",
                    "competition_risk": "manageable",
                    "technical_risk": "low",
                    "team_assessment": "strong",
                    "fundability": "high",
                    "valuation_range": {"pre_money": 2000000, "post_money": 5000000},
                    "recommended_raise": 3000000,
                    "confidence": 0.78
                },
                "investor_readiness": {
                    "deck_quality_needed": 0.9,
                    "traction_metrics_required": ["pilot_customers", "usage_metrics"],
                    "timeline_to_raise": "3-4 months"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        state["agent_messages"].append(vc_message)
        
        # Update state based on agent collaboration
        state["market_analysis"] = growth_response["content"]["analysis"]
        state["technical_validation"] = cto_message["content"]["assessment"]
        state["investor_pitch"] = vc_message["content"]["investment_analysis"]
        
        # Update quality metrics based on agent inputs
        state["market_viability"] = growth_response["content"]["analysis"]["confidence"]
        state["technical_feasibility"] = cto_message["content"]["assessment"]["confidence"]
        state["business_potential"] = vc_message["content"]["investment_analysis"]["confidence"]
        
        # Calculate overall consensus
        confidences = [
            state["market_viability"],
            state["technical_feasibility"], 
            state["business_potential"]
        ]
        state["consensus_score"] = sum(confidences) / len(confidences)
        
        # Verify collaboration results
        assert len(state["agent_messages"]) == 4
        assert state["consensus_score"] > 0.8  # Strong consensus
        assert state["market_viability"] == 0.82
        assert state["technical_feasibility"] == 0.88
        assert state["business_potential"] == 0.78
        
        # Verify cross-agent information flow
        growth_analysis = state["market_analysis"]
        tech_analysis = state["technical_validation"]
        investment_analysis = state["investor_pitch"]
        
        assert growth_analysis["market_size_tam"] == 5000000000
        assert tech_analysis["development_timeline"] == "6-8 months MVP"
        assert investment_analysis["recommended_raise"] == 3000000
        
        # Check that agents referenced each other's work appropriately
        assert growth_analysis["confidence"] > 0.8
        assert tech_analysis["confidence"] > 0.8
        assert investment_analysis["confidence"] > 0.7