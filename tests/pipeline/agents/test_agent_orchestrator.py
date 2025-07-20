"""
Comprehensive test suite for the Agent Orchestrator module.

Tests cover multi-agent workflow coordination, state machine transitions,
event handling, and performance metrics for the startup pipeline.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from pipeline.agents.agent_orchestrator import (
    AgentRole,
    WorkflowState,
    AgentCapability,
    AgentContext,
    AgentDecision,
    BaseAgent,
    CEOAgent,
    CTOAgent,
    VCAgent,
    StateTransition,
    AgentOrchestrator,
    get_agent_orchestrator
)
from pipeline.events.event_bus import DomainEvent, EventType


@pytest.fixture
def sample_agent_context():
    """Create a sample agent context for testing."""
    return AgentContext(
        current_state=WorkflowState.IDEATE,
        startup_idea="AI-powered healthcare monitoring platform",
        aggregate_id="test-idea-123",
        correlation_id="corr-456",
        execution_id="exec-789",
        budget_remaining=50.0
    )


@pytest.fixture
def sample_agent_decision():
    """Create a sample agent decision for testing."""
    return AgentDecision(
        agent_role=AgentRole.CEO,
        decision_type="strategic_analysis",
        decision_data={
            "market_size": "large",
            "confidence": 0.85,
            "reasoning": "Strong market opportunity"
        },
        confidence_score=0.85,
        reasoning="Comprehensive market analysis completed",
        next_state=WorkflowState.VALIDATE,
        required_agents=[AgentRole.CTO],
        execution_time=0.5,
        cost_estimate=0.15
    )


@pytest.fixture
def sample_domain_event():
    """Create a sample domain event for testing."""
    return DomainEvent(
        event_type=EventType.IDEA_CREATED,
        aggregate_id="test-idea-123",
        correlation_id="corr-456",
        event_data={
            "idea": "AI-powered healthcare monitoring platform",
            "title": "HealthTech AI Monitor",
            "description": "Predictive health monitoring using AI"
        }
    )


class TestAgentRole:
    """Test AgentRole enum."""
    
    def test_agent_role_values(self):
        """Test agent role enum values."""
        assert AgentRole.CEO.value == "ceo"
        assert AgentRole.CTO.value == "cto"
        assert AgentRole.VC.value == "vc"
        assert AgentRole.ANGEL.value == "angel"
        assert AgentRole.GROWTH.value == "growth"
        assert AgentRole.ORCHESTRATOR.value == "orchestrator"


class TestWorkflowState:
    """Test WorkflowState enum."""
    
    def test_workflow_state_values(self):
        """Test workflow state enum values."""
        assert WorkflowState.IDEATE.value == "ideate"
        assert WorkflowState.VALIDATE.value == "validate"
        assert WorkflowState.RESEARCH.value == "research"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.FAILED.value == "failed"
    
    def test_workflow_state_progression(self):
        """Test logical workflow state progression."""
        states = [
            WorkflowState.IDEATE,
            WorkflowState.VALIDATE,
            WorkflowState.RESEARCH,
            WorkflowState.DECK_GENERATION,
            WorkflowState.INVESTOR_EVALUATION,
            WorkflowState.SMOKE_TEST,
            WorkflowState.MVP_GENERATION,
            WorkflowState.DEPLOYMENT,
            WorkflowState.COMPLETED
        ]
        
        # Ensure all states are unique
        assert len(states) == len(set(states))


class TestAgentContext:
    """Test AgentContext data class."""
    
    def test_agent_context_creation(self, sample_agent_context):
        """Test agent context creation with required fields."""
        context = sample_agent_context
        
        assert context.current_state == WorkflowState.IDEATE
        assert context.startup_idea == "AI-powered healthcare monitoring platform"
        assert context.aggregate_id == "test-idea-123"
        assert context.correlation_id == "corr-456"
        assert isinstance(context.state_data, dict)
        assert isinstance(context.agent_messages, list)
        assert context.budget_remaining == 50.0
    
    def test_add_message(self, sample_agent_context):
        """Test inter-agent message functionality."""
        context = sample_agent_context
        
        # Add message from CEO to CTO
        message_content = {"analysis": "Market looks promising", "confidence": 0.8}
        context.add_message(AgentRole.CEO, AgentRole.CTO, message_content)
        
        assert len(context.agent_messages) == 1
        message = context.agent_messages[0]
        assert message['from'] == "ceo"
        assert message['to'] == "cto"
        assert message['content'] == message_content
        assert 'timestamp' in message
    
    def test_get_messages_for_agent(self, sample_agent_context):
        """Test filtering messages for specific agents."""
        context = sample_agent_context
        
        # Add multiple messages
        context.add_message(AgentRole.CEO, AgentRole.CTO, {"msg": "for_cto_1"})
        context.add_message(AgentRole.VC, AgentRole.CTO, {"msg": "for_cto_2"})
        context.add_message(AgentRole.CEO, AgentRole.VC, {"msg": "for_vc"})
        
        # Get messages for CTO
        cto_messages = context.get_messages_for_agent(AgentRole.CTO)
        assert len(cto_messages) == 2
        assert all(msg['to'] == "cto" for msg in cto_messages)
        
        # Get messages for VC
        vc_messages = context.get_messages_for_agent(AgentRole.VC)
        assert len(vc_messages) == 1
        assert vc_messages[0]['to'] == "vc"


class TestAgentDecision:
    """Test AgentDecision data class."""
    
    def test_agent_decision_creation(self, sample_agent_decision):
        """Test agent decision creation."""
        decision = sample_agent_decision
        
        assert decision.agent_role == AgentRole.CEO
        assert decision.decision_type == "strategic_analysis"
        assert decision.confidence_score == 0.85
        assert decision.next_state == WorkflowState.VALIDATE
        assert AgentRole.CTO in decision.required_agents
    
    def test_agent_decision_to_dict(self, sample_agent_decision):
        """Test agent decision serialization."""
        decision = sample_agent_decision
        decision_dict = decision.to_dict()
        
        assert decision_dict['agent_role'] == "ceo"
        assert decision_dict['decision_type'] == "strategic_analysis"
        assert decision_dict['confidence_score'] == 0.85
        assert decision_dict['next_state'] == "validate"
        assert "cto" in decision_dict['required_agents']
        assert decision_dict['execution_time'] == 0.5
        assert decision_dict['cost_estimate'] == 0.15


class TestBaseAgent:
    """Test BaseAgent abstract class functionality."""
    
    class MockAgent(BaseAgent):
        """Mock agent for testing."""
        
        async def execute(self, context: AgentContext) -> AgentDecision:
            return AgentDecision(
                agent_role=self.role,
                decision_type="mock_decision",
                decision_data={"test": True},
                confidence_score=0.9,
                reasoning="Mock execution"
            )
        
        def can_handle_state(self, state: WorkflowState) -> bool:
            return state == WorkflowState.IDEATE
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        return self.MockAgent(
            role=AgentRole.CEO,
            capabilities=[AgentCapability.STRATEGIC_PLANNING]
        )
    
    def test_base_agent_initialization(self, mock_agent):
        """Test base agent initialization."""
        agent = mock_agent
        
        assert agent.role == AgentRole.CEO
        assert AgentCapability.STRATEGIC_PLANNING in agent.capabilities
        assert agent.active is True
        assert agent.execution_count == 0
        assert agent.total_cost == 0.0
    
    @pytest.mark.asyncio
    async def test_base_agent_validate_input_success(self, mock_agent, sample_agent_context):
        """Test successful input validation."""
        result = await mock_agent.validate_input(sample_agent_context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_base_agent_validate_input_missing_idea(self, mock_agent):
        """Test input validation with missing startup idea."""
        context = AgentContext(
            current_state=WorkflowState.IDEATE,
            startup_idea="",  # Empty idea
            aggregate_id="test-123",
            correlation_id="corr-456"
        )
        
        result = await mock_agent.validate_input(context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_base_agent_validate_input_missing_aggregate_id(self, mock_agent):
        """Test input validation with missing aggregate ID."""
        context = AgentContext(
            current_state=WorkflowState.IDEATE,
            startup_idea="Valid idea",
            aggregate_id="",  # Empty aggregate ID
            correlation_id="corr-456"
        )
        
        result = await mock_agent.validate_input(context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, mock_agent):
        """Test agent metrics updating."""
        initial_count = mock_agent.execution_count
        initial_cost = mock_agent.total_cost
        
        await mock_agent.update_metrics(0.5, 0.15)
        
        assert mock_agent.execution_count == initial_count + 1
        assert mock_agent.total_cost == initial_cost + 0.15
        assert mock_agent.last_execution is not None


class TestCEOAgent:
    """Test CEO Agent functionality."""
    
    @pytest.fixture
    def ceo_agent(self):
        """Create CEO agent for testing."""
        return CEOAgent()
    
    def test_ceo_agent_initialization(self, ceo_agent):
        """Test CEO agent initialization."""
        assert ceo_agent.role == AgentRole.CEO
        assert AgentCapability.STRATEGIC_PLANNING in ceo_agent.capabilities
        assert AgentCapability.MARKET_RESEARCH in ceo_agent.capabilities
        assert AgentCapability.INVESTMENT_ANALYSIS in ceo_agent.capabilities
    
    def test_ceo_can_handle_state(self, ceo_agent):
        """Test which states CEO can handle."""
        assert ceo_agent.can_handle_state(WorkflowState.IDEATE) is True
        assert ceo_agent.can_handle_state(WorkflowState.VALIDATE) is True
        assert ceo_agent.can_handle_state(WorkflowState.INVESTOR_EVALUATION) is True
        assert ceo_agent.can_handle_state(WorkflowState.MVP_GENERATION) is False
        assert ceo_agent.can_handle_state(WorkflowState.DEPLOYMENT) is False
    
    @pytest.mark.asyncio
    async def test_ceo_execute_ideate_state(self, ceo_agent, sample_agent_context):
        """Test CEO execution in IDEATE state."""
        context = sample_agent_context
        context.current_state = WorkflowState.IDEATE
        
        decision = await ceo_agent.execute(context)
        
        assert decision.agent_role == AgentRole.CEO
        assert decision.decision_type == "strategic_analysis"
        assert decision.next_state == WorkflowState.VALIDATE
        assert AgentRole.CTO in decision.required_agents
        assert AgentRole.VP_RD in decision.required_agents
        assert decision.confidence_score > 0.0
        assert 'market_size' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_ceo_execute_validate_state(self, ceo_agent, sample_agent_context):
        """Test CEO execution in VALIDATE state."""
        context = sample_agent_context
        context.current_state = WorkflowState.VALIDATE
        
        decision = await ceo_agent.execute(context)
        
        assert decision.next_state == WorkflowState.RESEARCH
        assert AgentRole.GROWTH in decision.required_agents
        assert 'market_timing' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_ceo_execute_investor_evaluation_state(self, ceo_agent, sample_agent_context):
        """Test CEO execution in INVESTOR_EVALUATION state."""
        context = sample_agent_context
        context.current_state = WorkflowState.INVESTOR_EVALUATION
        
        decision = await ceo_agent.execute(context)
        
        assert decision.next_state == WorkflowState.SMOKE_TEST
        assert 'funding_readiness' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_ceo_execute_invalid_input(self, ceo_agent):
        """Test CEO execution with invalid input."""
        context = AgentContext(
            current_state=WorkflowState.IDEATE,
            startup_idea="",  # Invalid empty idea
            aggregate_id="test-123",
            correlation_id="corr-456"
        )
        
        decision = await ceo_agent.execute(context)
        
        assert decision.decision_type == "validation_failed"
        assert decision.confidence_score == 0.0


class TestCTOAgent:
    """Test CTO Agent functionality."""
    
    @pytest.fixture
    def cto_agent(self):
        """Create CTO agent for testing."""
        return CTOAgent()
    
    def test_cto_agent_initialization(self, cto_agent):
        """Test CTO agent initialization."""
        assert cto_agent.role == AgentRole.CTO
        assert AgentCapability.TECHNICAL_ARCHITECTURE in cto_agent.capabilities
        assert AgentCapability.MVP_CREATION in cto_agent.capabilities
        assert AgentCapability.PRODUCT_DEVELOPMENT in cto_agent.capabilities
    
    def test_cto_can_handle_state(self, cto_agent):
        """Test which states CTO can handle."""
        assert cto_agent.can_handle_state(WorkflowState.VALIDATE) is True
        assert cto_agent.can_handle_state(WorkflowState.MVP_GENERATION) is True
        assert cto_agent.can_handle_state(WorkflowState.DEPLOYMENT) is True
        assert cto_agent.can_handle_state(WorkflowState.IDEATE) is False
        assert cto_agent.can_handle_state(WorkflowState.INVESTOR_EVALUATION) is False
    
    @pytest.mark.asyncio
    async def test_cto_execute_validate_state(self, cto_agent, sample_agent_context):
        """Test CTO execution in VALIDATE state."""
        context = sample_agent_context
        context.current_state = WorkflowState.VALIDATE
        
        decision = await cto_agent.execute(context)
        
        assert decision.agent_role == AgentRole.CTO
        assert decision.decision_type == "technical_analysis"
        assert decision.next_state == WorkflowState.RESEARCH
        assert 'technical_complexity' in decision.decision_data
        assert 'technology_stack' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_cto_execute_mvp_generation_state(self, cto_agent, sample_agent_context):
        """Test CTO execution in MVP_GENERATION state."""
        context = sample_agent_context
        context.current_state = WorkflowState.MVP_GENERATION
        
        decision = await cto_agent.execute(context)
        
        assert decision.next_state == WorkflowState.DEPLOYMENT
        assert 'architecture_pattern' in decision.decision_data
        assert 'core_components' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_cto_execute_deployment_state(self, cto_agent, sample_agent_context):
        """Test CTO execution in DEPLOYMENT state."""
        context = sample_agent_context
        context.current_state = WorkflowState.DEPLOYMENT
        
        decision = await cto_agent.execute(context)
        
        assert decision.next_state == WorkflowState.COMPLETED
        assert 'deployment_platform' in decision.decision_data
        assert 'scaling_strategy' in decision.decision_data


class TestVCAgent:
    """Test VC Agent functionality."""
    
    @pytest.fixture
    def vc_agent(self):
        """Create VC agent for testing."""
        return VCAgent()
    
    def test_vc_agent_initialization(self, vc_agent):
        """Test VC agent initialization."""
        assert vc_agent.role == AgentRole.VC
        assert AgentCapability.INVESTMENT_ANALYSIS in vc_agent.capabilities
        assert AgentCapability.STRATEGIC_PLANNING in vc_agent.capabilities
        assert AgentCapability.MARKET_RESEARCH in vc_agent.capabilities
    
    def test_vc_can_handle_state(self, vc_agent):
        """Test which states VC can handle."""
        assert vc_agent.can_handle_state(WorkflowState.INVESTOR_EVALUATION) is True
        assert vc_agent.can_handle_state(WorkflowState.DECK_GENERATION) is True
        assert vc_agent.can_handle_state(WorkflowState.IDEATE) is False
        assert vc_agent.can_handle_state(WorkflowState.MVP_GENERATION) is False
    
    @pytest.mark.asyncio
    async def test_vc_execute_investor_evaluation_state(self, vc_agent, sample_agent_context):
        """Test VC execution in INVESTOR_EVALUATION state."""
        context = sample_agent_context
        context.current_state = WorkflowState.INVESTOR_EVALUATION
        
        decision = await vc_agent.execute(context)
        
        assert decision.agent_role == AgentRole.VC
        assert decision.decision_type == "investment_analysis"
        assert decision.next_state == WorkflowState.DECK_GENERATION
        assert 'investment_thesis' in decision.decision_data
        assert 'market_size_tam' in decision.decision_data
    
    @pytest.mark.asyncio
    async def test_vc_execute_deck_generation_state(self, vc_agent, sample_agent_context):
        """Test VC execution in DECK_GENERATION state."""
        context = sample_agent_context
        context.current_state = WorkflowState.DECK_GENERATION
        
        decision = await vc_agent.execute(context)
        
        assert decision.next_state == WorkflowState.SMOKE_TEST
        assert 'key_slides_required' in decision.decision_data
        assert 'narrative_focus' in decision.decision_data


class TestStateTransition:
    """Test StateTransition data class."""
    
    def test_state_transition_creation(self):
        """Test state transition creation."""
        transition = StateTransition(
            from_state=WorkflowState.IDEATE,
            to_state=WorkflowState.VALIDATE,
            required_agents=[AgentRole.CEO, AgentRole.CTO],
            conditions={'confidence_threshold': 0.7},
            timeout_seconds=300
        )
        
        assert transition.from_state == WorkflowState.IDEATE
        assert transition.to_state == WorkflowState.VALIDATE
        assert AgentRole.CEO in transition.required_agents
        assert AgentRole.CTO in transition.required_agents
        assert transition.conditions['confidence_threshold'] == 0.7
        assert transition.timeout_seconds == 300


class TestAgentOrchestrator:
    """Test Agent Orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        # Reset singleton
        import pipeline.agents.agent_orchestrator
        pipeline.agents.agent_orchestrator._agent_orchestrator = None
        
        # Create new orchestrator with mocked dependencies
        with patch('pipeline.agents.agent_orchestrator.get_settings') as mock_settings, \
             patch('pipeline.agents.agent_orchestrator.get_event_bus') as mock_event_bus:
            
            mock_settings.return_value = Mock()
            mock_event_bus.return_value = Mock()
            
            orchestrator = AgentOrchestrator()
            return orchestrator
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.agents) == 3  # CEO, CTO, VC
        assert AgentRole.CEO in orchestrator.agents
        assert AgentRole.CTO in orchestrator.agents
        assert AgentRole.VC in orchestrator.agents
        
        assert len(orchestrator.state_transitions) > 0
        assert orchestrator.execution_metrics['total_workflows'] == 0
        assert orchestrator.execution_metrics['completed_workflows'] == 0
        assert orchestrator.execution_metrics['failed_workflows'] == 0
    
    def test_orchestrator_handled_events(self, orchestrator):
        """Test orchestrator event handling registration."""
        handled_events = orchestrator.handled_events
        
        assert EventType.IDEA_CREATED in handled_events
        assert EventType.IDEA_VALIDATED in handled_events
        assert EventType.PROCESSING_COMPLETE in handled_events
        assert EventType.WORKFLOW_STATE_CHANGED in handled_events
    
    @pytest.mark.asyncio
    async def test_start_workflow(self, orchestrator, sample_domain_event):
        """Test starting a new workflow."""
        with patch.object(orchestrator, '_execute_workflow_state', new_callable=AsyncMock) as mock_execute:
            await orchestrator._start_workflow(sample_domain_event)
            
            # Check workflow was created
            assert sample_domain_event.aggregate_id in orchestrator.active_workflows
            context = orchestrator.active_workflows[sample_domain_event.aggregate_id]
            assert context.current_state == WorkflowState.IDEATE
            assert context.startup_idea == "AI-powered healthcare monitoring platform"
            assert orchestrator.execution_metrics['total_workflows'] == 1
            
            # Check execution was triggered
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_workflow_missing_idea(self, orchestrator):
        """Test starting workflow with missing idea."""
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            aggregate_id="test-123",
            correlation_id="corr-456",
            event_data={}  # Missing idea
        )
        
        await orchestrator._start_workflow(event)
        
        # Should not create workflow
        assert "test-123" not in orchestrator.active_workflows
        assert orchestrator.execution_metrics['total_workflows'] == 0
    
    @pytest.mark.asyncio
    async def test_execute_workflow_state_success(self, orchestrator, sample_agent_context):
        """Test successful workflow state execution."""
        # Add context to active workflows
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        with patch.object(orchestrator, '_process_agent_decisions', new_callable=AsyncMock) as mock_process:
            await orchestrator._execute_workflow_state(sample_agent_context)
            
            # Should process agent decisions
            mock_process.assert_called_once()
            args = mock_process.call_args[0]
            assert args[0] == sample_agent_context
            assert len(args[1]) > 0  # Should have agent decisions
    
    @pytest.mark.asyncio
    async def test_execute_workflow_state_no_agents(self, orchestrator, sample_agent_context):
        """Test workflow state execution with no available agents."""
        # Set state that no agents can handle
        sample_agent_context.current_state = WorkflowState.RESEARCH
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        with patch.object(orchestrator, '_transition_to_failed_state', new_callable=AsyncMock) as mock_fail:
            await orchestrator._execute_workflow_state(sample_agent_context)
            
            # Should transition to failed state
            mock_fail.assert_called_once_with(sample_agent_context, "No available agents")
    
    @pytest.mark.asyncio
    async def test_process_agent_decisions_consensus(self, orchestrator, sample_agent_context):
        """Test processing agent decisions with consensus."""
        decisions = [
            AgentDecision(
                agent_role=AgentRole.CEO,
                decision_type="test",
                decision_data={"confidence": 0.8},
                confidence_score=0.8,
                reasoning="Test",
                next_state=WorkflowState.VALIDATE
            ),
            AgentDecision(
                agent_role=AgentRole.CTO,
                decision_type="test",
                decision_data={"confidence": 0.9},
                confidence_score=0.9,
                reasoning="Test",
                next_state=WorkflowState.VALIDATE
            )
        ]
        
        with patch.object(orchestrator, '_transition_to_state', new_callable=AsyncMock) as mock_transition:
            await orchestrator._process_agent_decisions(sample_agent_context, decisions)
            
            # Should transition to consensus state
            mock_transition.assert_called_once_with(sample_agent_context, WorkflowState.VALIDATE)
    
    @pytest.mark.asyncio
    async def test_process_agent_decisions_low_confidence(self, orchestrator, sample_agent_context):
        """Test processing agent decisions with low confidence."""
        decisions = [
            AgentDecision(
                agent_role=AgentRole.CEO,
                decision_type="test",
                decision_data={"confidence": 0.5},
                confidence_score=0.5,  # Low confidence
                reasoning="Test",
                next_state=WorkflowState.VALIDATE
            )
        ]
        
        with patch.object(orchestrator, '_execute_workflow_state', new_callable=AsyncMock) as mock_execute:
            await orchestrator._process_agent_decisions(sample_agent_context, decisions)
            
            # Should retry execution
            mock_execute.assert_called_once()
            assert sample_agent_context.state_data['retry_count'] == 1
    
    def test_determine_consensus_next_state(self, orchestrator):
        """Test consensus determination from agent decisions."""
        decisions = [
            AgentDecision(
                agent_role=AgentRole.CEO,
                decision_type="test",
                decision_data={},
                confidence_score=0.8,
                reasoning="Test",
                next_state=WorkflowState.VALIDATE
            ),
            AgentDecision(
                agent_role=AgentRole.CTO,
                decision_type="test",
                decision_data={},
                confidence_score=0.9,
                reasoning="Test",
                next_state=WorkflowState.VALIDATE
            ),
            AgentDecision(
                agent_role=AgentRole.VC,
                decision_type="test",
                decision_data={},
                confidence_score=0.7,
                reasoning="Test",
                next_state=WorkflowState.RESEARCH
            )
        ]
        
        consensus = orchestrator._determine_consensus_next_state(decisions)
        assert consensus == WorkflowState.VALIDATE  # Majority vote
    
    def test_determine_consensus_no_states(self, orchestrator):
        """Test consensus determination with no next states."""
        decisions = [
            AgentDecision(
                agent_role=AgentRole.CEO,
                decision_type="test",
                decision_data={},
                confidence_score=0.8,
                reasoning="Test",
                next_state=None  # No next state
            )
        ]
        
        consensus = orchestrator._determine_consensus_next_state(decisions)
        assert consensus is None
    
    @pytest.mark.asyncio
    async def test_transition_to_state(self, orchestrator, sample_agent_context):
        """Test state transition functionality."""
        old_state = sample_agent_context.current_state
        new_state = WorkflowState.VALIDATE
        
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        with patch.object(orchestrator.event_bus, 'publish', new_callable=AsyncMock) as mock_publish, \
             patch.object(orchestrator, '_execute_workflow_state', new_callable=AsyncMock) as mock_execute:
            
            await orchestrator._transition_to_state(sample_agent_context, new_state)
            
            # Check state was updated
            assert sample_agent_context.current_state == new_state
            
            # Check event was published
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.event_type == EventType.WORKFLOW_STATE_CHANGED
            assert event.event_data['old_state'] == old_state.value
            assert event.event_data['new_state'] == new_state.value
            
            # Check execution was triggered
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transition_to_completed_state(self, orchestrator, sample_agent_context):
        """Test transition to completed state."""
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        with patch.object(orchestrator, '_complete_workflow', new_callable=AsyncMock) as mock_complete:
            await orchestrator._transition_to_state(sample_agent_context, WorkflowState.COMPLETED)
            
            # Should complete workflow instead of executing
            mock_complete.assert_called_once_with(sample_agent_context)
    
    @pytest.mark.asyncio
    async def test_transition_to_failed_state(self, orchestrator, sample_agent_context):
        """Test transition to failed state."""
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        initial_failed_count = orchestrator.execution_metrics['failed_workflows']
        
        with patch.object(orchestrator.event_bus, 'publish', new_callable=AsyncMock) as mock_publish:
            await orchestrator._transition_to_failed_state(sample_agent_context, "Test failure")
            
            # Check state and metrics updated
            assert sample_agent_context.current_state == WorkflowState.FAILED
            assert sample_agent_context.state_data['failure_reason'] == "Test failure"
            assert orchestrator.execution_metrics['failed_workflows'] == initial_failed_count + 1
            
            # Check event was published
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.event_type == EventType.PIPELINE_FAILED
            
            # Check workflow was cleaned up
            assert sample_agent_context.aggregate_id not in orchestrator.active_workflows
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, orchestrator, sample_agent_context):
        """Test workflow completion."""
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        initial_completed_count = orchestrator.execution_metrics['completed_workflows']
        
        with patch.object(orchestrator.event_bus, 'publish', new_callable=AsyncMock) as mock_publish:
            await orchestrator._complete_workflow(sample_agent_context)
            
            # Check metrics updated
            assert orchestrator.execution_metrics['completed_workflows'] == initial_completed_count + 1
            
            # Check event was published
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.event_type == EventType.OUTPUT_COMPLETE
            
            # Check workflow was cleaned up
            assert sample_agent_context.aggregate_id not in orchestrator.active_workflows
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, orchestrator, sample_agent_context):
        """Test getting workflow status."""
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        status = await orchestrator.get_workflow_status(sample_agent_context.aggregate_id)
        
        assert status is not None
        assert status['aggregate_id'] == sample_agent_context.aggregate_id
        assert status['current_state'] == sample_agent_context.current_state.value
        assert status['startup_idea'] == sample_agent_context.startup_idea
        assert status['budget_remaining'] == sample_agent_context.budget_remaining
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self, orchestrator):
        """Test getting status for non-existent workflow."""
        status = await orchestrator.get_workflow_status("non-existent-id")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_orchestrator_metrics(self, orchestrator, sample_agent_context):
        """Test getting orchestrator metrics."""
        orchestrator.active_workflows[sample_agent_context.aggregate_id] = sample_agent_context
        
        metrics = await orchestrator.get_orchestrator_metrics()
        
        assert 'total_workflows' in metrics
        assert 'completed_workflows' in metrics
        assert 'failed_workflows' in metrics
        assert 'active_workflows' in metrics
        assert 'registered_agents' in metrics
        assert 'state_transitions' in metrics
        
        assert metrics['active_workflows'] == 1
        assert metrics['registered_agents'] == 3
        assert metrics['state_transitions'] > 0


class TestAgentOrchestratorSingleton:
    """Test Agent Orchestrator singleton functionality."""
    
    def test_get_agent_orchestrator_singleton(self):
        """Test singleton pattern for agent orchestrator."""
        # Reset singleton
        import pipeline.agents.agent_orchestrator
        pipeline.agents.agent_orchestrator._agent_orchestrator = None
        
        with patch('pipeline.agents.agent_orchestrator.get_settings') as mock_settings, \
             patch('pipeline.agents.agent_orchestrator.get_event_bus') as mock_event_bus:
            
            mock_settings.return_value = Mock()
            mock_event_bus.return_value = Mock()
            
            # Get orchestrator instances
            orchestrator1 = get_agent_orchestrator()
            orchestrator2 = get_agent_orchestrator()
            
            # Should be the same instance
            assert orchestrator1 is orchestrator2
            assert isinstance(orchestrator1, AgentOrchestrator)


class TestAgentOrchestratorIntegration:
    """Integration tests for Agent Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution_simulation(self):
        """Test simulated full workflow execution."""
        # Reset singleton
        import pipeline.agents.agent_orchestrator
        pipeline.agents.agent_orchestrator._agent_orchestrator = None
        
        with patch('pipeline.agents.agent_orchestrator.get_settings') as mock_settings, \
             patch('pipeline.agents.agent_orchestrator.get_event_bus') as mock_event_bus:
            
            mock_settings.return_value = Mock()
            mock_event_bus.return_value = Mock()
            
            orchestrator = AgentOrchestrator()
            
            # Create idea event
            event = DomainEvent(
                event_type=EventType.IDEA_CREATED,
                aggregate_id="integration-test-123",
                correlation_id="corr-integration",
                event_data={
                    "idea": "AI-powered code review assistant",
                    "title": "CodeReview AI",
                    "description": "Automated intelligent code review platform"
                }
            )
            
            # Start workflow
            await orchestrator._start_workflow(event)
            
            # Verify workflow was created
            assert "integration-test-123" in orchestrator.active_workflows
            context = orchestrator.active_workflows["integration-test-123"]
            assert context.current_state == WorkflowState.IDEATE
            assert "AI-powered code review assistant" in context.startup_idea
            
            # Verify metrics
            assert orchestrator.execution_metrics['total_workflows'] == 1
            
            # Get workflow status
            status = await orchestrator.get_workflow_status("integration-test-123")
            assert status is not None
            assert status['aggregate_id'] == "integration-test-123"
    
    @pytest.mark.asyncio
    async def test_agent_communication_flow(self):
        """Test inter-agent communication flow."""
        context = AgentContext(
            current_state=WorkflowState.IDEATE,
            startup_idea="Blockchain-based supply chain tracker",
            aggregate_id="comm-test-123",
            correlation_id="corr-comm"
        )
        
        # CEO adds message for CTO
        ceo_message = {
            "market_analysis": "Strong demand in logistics sector",
            "recommended_focus": "B2B enterprise customers",
            "confidence": 0.85
        }
        context.add_message(AgentRole.CEO, AgentRole.CTO, ceo_message)
        
        # CTO adds response for CEO
        cto_response = {
            "technical_feasibility": "High - well-established blockchain frameworks",
            "development_timeline": "6-8 months for MVP",
            "recommended_stack": ["ethereum", "react", "node.js"]
        }
        context.add_message(AgentRole.CTO, AgentRole.CEO, cto_response)
        
        # VC adds investment perspective
        vc_message = {
            "market_opportunity": "$50B+ logistics technology market",
            "investment_readiness": "requires technical prototype first",
            "funding_recommendation": "$500K seed round"
        }
        context.add_message(AgentRole.VC, AgentRole.CEO, vc_message)
        
        # Verify message filtering
        cto_messages = context.get_messages_for_agent(AgentRole.CTO)
        ceo_messages = context.get_messages_for_agent(AgentRole.CEO)
        
        assert len(cto_messages) == 1
        assert len(ceo_messages) == 2
        
        assert cto_messages[0]['content'] == ceo_message
        assert any(msg['content'] == cto_response for msg in ceo_messages)
        assert any(msg['content'] == vc_message for msg in ceo_messages)


# Performance and stress testing
class TestAgentOrchestratorPerformance:
    """Performance tests for Agent Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_handling(self):
        """Test handling multiple concurrent workflows."""
        # Reset singleton
        import pipeline.agents.agent_orchestrator
        pipeline.agents.agent_orchestrator._agent_orchestrator = None
        
        with patch('pipeline.agents.agent_orchestrator.get_settings') as mock_settings, \
             patch('pipeline.agents.agent_orchestrator.get_event_bus') as mock_event_bus:
            
            mock_settings.return_value = Mock()
            mock_event_bus.return_value = Mock()
            
            orchestrator = AgentOrchestrator()
            
            # Create multiple workflow events
            events = []
            for i in range(10):
                event = DomainEvent(
                    event_type=EventType.IDEA_CREATED,
                    aggregate_id=f"perf-test-{i}",
                    correlation_id=f"corr-perf-{i}",
                    event_data={
                        "idea": f"Performance test idea {i}",
                        "title": f"PerfTest {i}",
                        "description": f"Performance testing idea number {i}"
                    }
                )
                events.append(event)
            
            # Start all workflows concurrently
            start_time = asyncio.get_event_loop().time()
            await asyncio.gather(*[orchestrator._start_workflow(event) for event in events])
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Verify all workflows were created
            assert len(orchestrator.active_workflows) == 10
            assert orchestrator.execution_metrics['total_workflows'] == 10
            
            # Performance assertion (should complete within reasonable time)
            assert execution_time < 5.0  # 5 seconds max for 10 workflows
            
            # Verify all workflows have correct initial state
            for i in range(10):
                workflow_id = f"perf-test-{i}"
                assert workflow_id in orchestrator.active_workflows
                context = orchestrator.active_workflows[workflow_id]
                assert context.current_state == WorkflowState.IDEATE