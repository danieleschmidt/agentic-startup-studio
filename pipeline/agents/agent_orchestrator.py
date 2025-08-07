"""
Agent Orchestrator - LangGraph-based multi-agent coordination system.

Manages the workflow state machine and coordinates specialized AI agents
for CEO, CTO, VC, Angel, and Growth roles throughout the startup pipeline.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pipeline.config.settings import get_settings
from pipeline.events.event_bus import (
    DomainEvent,
    EventHandler,
    EventType,
    get_event_bus,
)

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Specialized agent roles in the startup pipeline."""
    CEO = "ceo"
    CTO = "cto"
    VP_RD = "vp_rd"
    VC = "vc"
    ANGEL = "angel"
    GROWTH = "growth"
    ORCHESTRATOR = "orchestrator"


class WorkflowState(Enum):
    """LangGraph workflow states for the startup pipeline."""
    IDEATE = "ideate"
    VALIDATE = "validate"
    RESEARCH = "research"
    DECK_GENERATION = "deck_generation"
    INVESTOR_EVALUATION = "investor_evaluation"
    SMOKE_TEST = "smoke_test"
    MVP_GENERATION = "mvp_generation"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentCapability(Enum):
    """Agent capabilities and tool access."""
    STRATEGIC_PLANNING = "strategic_planning"
    TECHNICAL_ARCHITECTURE = "technical_architecture"
    PRODUCT_DEVELOPMENT = "product_development"
    INVESTMENT_ANALYSIS = "investment_analysis"
    MARKET_RESEARCH = "market_research"
    GROWTH_STRATEGY = "growth_strategy"
    EVIDENCE_COLLECTION = "evidence_collection"
    DECK_GENERATION = "deck_generation"
    MVP_CREATION = "mvp_creation"


@dataclass
class AgentContext:
    """Context passed to agents during execution."""
    current_state: WorkflowState
    startup_idea: str
    aggregate_id: str
    correlation_id: str

    # State data
    state_data: dict[str, Any] = field(default_factory=dict)

    # Agent communication
    agent_messages: list[dict[str, Any]] = field(default_factory=list)

    # Execution metadata
    execution_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Budget and constraints
    budget_remaining: float = 0.0
    max_execution_time: int = 3600  # seconds

    def add_message(self, from_agent: AgentRole, to_agent: AgentRole, content: dict[str, Any]) -> None:
        """Add inter-agent communication message."""
        self.agent_messages.append({
            'from': from_agent.value,
            'to': to_agent.value,
            'content': content,
            'timestamp': datetime.now(UTC).isoformat()
        })

    def get_messages_for_agent(self, agent_role: AgentRole) -> list[dict[str, Any]]:
        """Get messages directed to a specific agent."""
        return [
            msg for msg in self.agent_messages
            if msg['to'] == agent_role.value
        ]


@dataclass
class AgentDecision:
    """Decision output from an agent."""
    agent_role: AgentRole
    decision_type: str
    decision_data: dict[str, Any]
    confidence_score: float
    reasoning: str

    # Next state recommendation
    next_state: WorkflowState | None = None
    required_agents: list[AgentRole] = field(default_factory=list)

    # Quality metrics
    execution_time: float = 0.0
    cost_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        decision_dict = asdict(self)
        decision_dict['agent_role'] = self.agent_role.value
        if self.next_state:
            decision_dict['next_state'] = self.next_state.value
        decision_dict['required_agents'] = [agent.value for agent in self.required_agents]
        return decision_dict


class BaseAgent(ABC):
    """Abstract base class for specialized agents."""

    def __init__(self, role: AgentRole, capabilities: list[AgentCapability]):
        self.role = role
        self.capabilities = capabilities
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{__name__}.{role.value}")

        # Agent state
        self.active = True
        self.last_execution = None
        self.execution_count = 0
        self.total_cost = 0.0

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute agent logic and return decision."""
        pass

    @abstractmethod
    def can_handle_state(self, state: WorkflowState) -> bool:
        """Check if agent can handle the given workflow state."""
        pass

    async def validate_input(self, context: AgentContext) -> bool:
        """Validate input context before execution."""
        if not context.startup_idea:
            self.logger.error("Missing startup idea in context")
            return False

        if not context.aggregate_id:
            self.logger.error("Missing aggregate ID in context")
            return False

        return True

    async def update_metrics(self, execution_time: float, cost: float) -> None:
        """Update agent execution metrics."""
        self.execution_count += 1
        self.total_cost += cost
        self.last_execution = datetime.now(UTC)

        self.logger.debug(
            f"Agent {self.role.value} metrics updated",
            extra={
                'execution_count': self.execution_count,
                'total_cost': self.total_cost,
                'execution_time': execution_time
            }
        )


class CEOAgent(BaseAgent):
    """CEO Agent - Strategic vision and business validation."""

    def __init__(self):
        super().__init__(
            role=AgentRole.CEO,
            capabilities=[
                AgentCapability.STRATEGIC_PLANNING,
                AgentCapability.MARKET_RESEARCH,
                AgentCapability.INVESTMENT_ANALYSIS
            ]
        )

    def can_handle_state(self, state: WorkflowState) -> bool:
        """CEO handles strategic states."""
        return state in [
            WorkflowState.IDEATE,
            WorkflowState.VALIDATE,
            WorkflowState.INVESTOR_EVALUATION
        ]

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute CEO strategic decision-making."""
        start_time = asyncio.get_event_loop().time()

        if not await self.validate_input(context):
            return AgentDecision(
                agent_role=self.role,
                decision_type="validation_failed",
                decision_data={},
                confidence_score=0.0,
                reasoning="Input validation failed"
            )

        self.logger.info(f"CEO analyzing startup idea: {context.startup_idea[:100]}...")

        # Strategic analysis based on current state
        if context.current_state == WorkflowState.IDEATE:
            decision_data = await self._analyze_idea_feasibility(context)
            next_state = WorkflowState.VALIDATE
            required_agents = [AgentRole.CTO, AgentRole.VP_RD]

        elif context.current_state == WorkflowState.VALIDATE:
            decision_data = await self._validate_market_opportunity(context)
            next_state = WorkflowState.RESEARCH
            required_agents = [AgentRole.GROWTH]

        elif context.current_state == WorkflowState.INVESTOR_EVALUATION:
            decision_data = await self._evaluate_investment_readiness(context)
            next_state = WorkflowState.SMOKE_TEST
            required_agents = [AgentRole.GROWTH, AgentRole.CTO]

        else:
            decision_data = {"error": "Unsupported state for CEO agent"}
            next_state = None
            required_agents = []

        execution_time = asyncio.get_event_loop().time() - start_time
        cost_estimate = 0.15  # Estimated GPT-4 cost

        await self.update_metrics(execution_time, cost_estimate)

        return AgentDecision(
            agent_role=self.role,
            decision_type="strategic_analysis",
            decision_data=decision_data,
            confidence_score=decision_data.get('confidence', 0.8),
            reasoning=decision_data.get('reasoning', 'Strategic analysis completed'),
            next_state=next_state,
            required_agents=required_agents,
            execution_time=execution_time,
            cost_estimate=cost_estimate
        )

    async def _analyze_idea_feasibility(self, context: AgentContext) -> dict[str, Any]:
        """Analyze startup idea feasibility from CEO perspective."""
        # Simulated CEO analysis - in real implementation would use LLM
        return {
            'market_size': 'large',
            'competitive_landscape': 'moderate',
            'business_model_clarity': 'high',
            'scalability_potential': 'high',
            'confidence': 0.85,
            'reasoning': 'Strong business fundamentals with clear value proposition',
            'key_risks': ['market timing', 'competition'],
            'success_factors': ['execution speed', 'product-market fit']
        }

    async def _validate_market_opportunity(self, context: AgentContext) -> dict[str, Any]:
        """Validate market opportunity and timing."""
        return {
            'market_timing': 'optimal',
            'customer_demand': 'high',
            'regulatory_environment': 'favorable',
            'confidence': 0.78,
            'reasoning': 'Market conditions support launch timing',
            'go_to_market_strategy': 'direct sales + digital marketing',
            'revenue_projection': '6-month break-even possible'
        }

    async def _evaluate_investment_readiness(self, context: AgentContext) -> dict[str, Any]:
        """Evaluate readiness for investment."""
        return {
            'funding_readiness': 'high',
            'valuation_estimate': '$2-5M pre-money',
            'investor_fit': 'seed/series-a',
            'confidence': 0.82,
            'reasoning': 'Strong fundamentals and clear growth path',
            'recommended_raise': '$500K-1M',
            'runway_months': 18
        }


class CTOAgent(BaseAgent):
    """CTO Agent - Technical architecture and implementation."""

    def __init__(self):
        super().__init__(
            role=AgentRole.CTO,
            capabilities=[
                AgentCapability.TECHNICAL_ARCHITECTURE,
                AgentCapability.MVP_CREATION,
                AgentCapability.PRODUCT_DEVELOPMENT
            ]
        )

    def can_handle_state(self, state: WorkflowState) -> bool:
        """CTO handles technical states."""
        return state in [
            WorkflowState.VALIDATE,
            WorkflowState.MVP_GENERATION,
            WorkflowState.DEPLOYMENT
        ]

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute CTO technical decision-making."""
        start_time = asyncio.get_event_loop().time()

        if not await self.validate_input(context):
            return AgentDecision(
                agent_role=self.role,
                decision_type="validation_failed",
                decision_data={},
                confidence_score=0.0,
                reasoning="Input validation failed"
            )

        self.logger.info(f"CTO analyzing technical requirements for: {context.startup_idea[:100]}...")

        # Technical analysis based on current state
        if context.current_state == WorkflowState.VALIDATE:
            decision_data = await self._analyze_technical_feasibility(context)
            next_state = WorkflowState.RESEARCH
            required_agents = [AgentRole.VP_RD]

        elif context.current_state == WorkflowState.MVP_GENERATION:
            decision_data = await self._design_mvp_architecture(context)
            next_state = WorkflowState.DEPLOYMENT
            required_agents = [AgentRole.VP_RD]

        elif context.current_state == WorkflowState.DEPLOYMENT:
            decision_data = await self._plan_deployment_strategy(context)
            next_state = WorkflowState.COMPLETED
            required_agents = []

        else:
            decision_data = {"error": "Unsupported state for CTO agent"}
            next_state = None
            required_agents = []

        execution_time = asyncio.get_event_loop().time() - start_time
        cost_estimate = 0.12  # Estimated cost

        await self.update_metrics(execution_time, cost_estimate)

        return AgentDecision(
            agent_role=self.role,
            decision_type="technical_analysis",
            decision_data=decision_data,
            confidence_score=decision_data.get('confidence', 0.85),
            reasoning=decision_data.get('reasoning', 'Technical analysis completed'),
            next_state=next_state,
            required_agents=required_agents,
            execution_time=execution_time,
            cost_estimate=cost_estimate
        )

    async def _analyze_technical_feasibility(self, context: AgentContext) -> dict[str, Any]:
        """Analyze technical feasibility and architecture requirements."""
        return {
            'technical_complexity': 'medium',
            'development_time_estimate': '3-4 months',
            'technology_stack': ['python', 'react', 'postgresql'],
            'scalability_requirements': 'moderate',
            'security_considerations': ['data encryption', 'user authentication'],
            'confidence': 0.88,
            'reasoning': 'Well-understood technology stack with proven scalability',
            'key_technical_risks': ['third-party integrations', 'data volume'],
            'mitigation_strategies': ['phased rollout', 'performance monitoring']
        }

    async def _design_mvp_architecture(self, context: AgentContext) -> dict[str, Any]:
        """Design MVP architecture and implementation plan."""
        return {
            'architecture_pattern': 'microservices',
            'core_components': ['api', 'frontend', 'database', 'auth'],
            'deployment_strategy': 'containerized',
            'monitoring_stack': ['prometheus', 'grafana'],
            'confidence': 0.92,
            'reasoning': 'Modular architecture supports rapid iteration',
            'development_phases': ['core features', 'user interface', 'integrations'],
            'resource_requirements': '2-3 developers, 4-6 weeks'
        }

    async def _plan_deployment_strategy(self, context: AgentContext) -> dict[str, Any]:
        """Plan deployment and infrastructure strategy."""
        return {
            'deployment_platform': 'fly.io',
            'infrastructure_cost': '$50-100/month',
            'scaling_strategy': 'horizontal',
            'backup_strategy': 'automated daily',
            'monitoring_alerts': 'enabled',
            'confidence': 0.90,
            'reasoning': 'Proven deployment stack with cost-effective scaling',
            'rollback_plan': 'blue-green deployment',
            'performance_targets': '< 200ms response time'
        }


class VCAgent(BaseAgent):
    """VC Agent - Investment analysis and due diligence."""

    def __init__(self):
        super().__init__(
            role=AgentRole.VC,
            capabilities=[
                AgentCapability.INVESTMENT_ANALYSIS,
                AgentCapability.STRATEGIC_PLANNING,
                AgentCapability.MARKET_RESEARCH
            ]
        )

    def can_handle_state(self, state: WorkflowState) -> bool:
        """VC handles investment-related states."""
        return state in [
            WorkflowState.INVESTOR_EVALUATION,
            WorkflowState.DECK_GENERATION
        ]

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute VC investment analysis."""
        start_time = asyncio.get_event_loop().time()

        if not await self.validate_input(context):
            return AgentDecision(
                agent_role=self.role,
                decision_type="validation_failed",
                decision_data={},
                confidence_score=0.0,
                reasoning="Input validation failed"
            )

        self.logger.info(f"VC evaluating investment opportunity: {context.startup_idea[:100]}...")

        if context.current_state == WorkflowState.INVESTOR_EVALUATION:
            decision_data = await self._evaluate_investment_opportunity(context)
            next_state = WorkflowState.DECK_GENERATION
            required_agents = [AgentRole.CEO]

        elif context.current_state == WorkflowState.DECK_GENERATION:
            decision_data = await self._review_pitch_deck_requirements(context)
            next_state = WorkflowState.SMOKE_TEST
            required_agents = [AgentRole.GROWTH]

        else:
            decision_data = {"error": "Unsupported state for VC agent"}
            next_state = None
            required_agents = []

        execution_time = asyncio.get_event_loop().time() - start_time
        cost_estimate = 0.10

        await self.update_metrics(execution_time, cost_estimate)

        return AgentDecision(
            agent_role=self.role,
            decision_type="investment_analysis",
            decision_data=decision_data,
            confidence_score=decision_data.get('confidence', 0.75),
            reasoning=decision_data.get('reasoning', 'Investment analysis completed'),
            next_state=next_state,
            required_agents=required_agents,
            execution_time=execution_time,
            cost_estimate=cost_estimate
        )

    async def _evaluate_investment_opportunity(self, context: AgentContext) -> dict[str, Any]:
        """Evaluate investment opportunity from VC perspective."""
        return {
            'investment_thesis': 'strong market opportunity with experienced team',
            'market_size_tam': '$10B+',
            'competitive_moat': 'technology and first-mover advantage',
            'team_assessment': 'experienced founders with domain expertise',
            'traction_metrics': 'early validation signals positive',
            'confidence': 0.78,
            'reasoning': 'Solid fundamentals with clear path to scale',
            'risk_factors': ['market timing', 'execution risk', 'competition'],
            'investment_recommendation': 'proceed with due diligence'
        }

    async def _review_pitch_deck_requirements(self, context: AgentContext) -> dict[str, Any]:
        """Review and specify pitch deck requirements."""
        return {
            'key_slides_required': [
                'problem_statement',
                'solution_overview',
                'market_opportunity',
                'business_model',
                'traction',
                'financial_projections',
                'team',
                'funding_ask'
            ],
            'narrative_focus': 'problem-solution fit and market opportunity',
            'data_requirements': 'market size, early traction metrics',
            'confidence': 0.85,
            'reasoning': 'Standard VC deck format with startup-specific focus',
            'presentation_style': 'data-driven with clear value proposition',
            'expected_meeting_outcome': '15-20% interest rate for follow-up'
        }


@dataclass
class StateTransition:
    """Workflow state transition configuration."""
    from_state: WorkflowState
    to_state: WorkflowState
    required_agents: list[AgentRole]
    conditions: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300


class AgentOrchestrator(EventHandler):
    """
    Agent Orchestrator - Coordinates multi-agent workflow execution.
    
    Manages LangGraph state machine, agent coordination, and event-driven
    transitions between workflow states.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.event_bus = get_event_bus()

        # Agent registry
        self.agents: dict[AgentRole, BaseAgent] = {}
        self._initialize_agents()

        # Workflow state management
        self.active_workflows: dict[str, AgentContext] = {}  # aggregate_id -> context
        self.state_transitions = self._initialize_state_machine()

        # Execution tracking
        self.execution_metrics = {
            'total_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0,
            'total_cost': 0.0
        }

        # Subscribe to relevant events
        self.event_bus.subscribe([
            EventType.IDEA_CREATED,
            EventType.IDEA_VALIDATED,
            EventType.PROCESSING_COMPLETE,
            EventType.TRANSFORMATION_COMPLETE,
            EventType.OUTPUT_COMPLETE,
            EventType.WORKFLOW_STATE_CHANGED
        ], self)

    @property
    def handled_events(self) -> list[EventType]:
        """Events handled by the orchestrator."""
        return [
            EventType.IDEA_CREATED,
            EventType.IDEA_VALIDATED,
            EventType.PROCESSING_COMPLETE,
            EventType.TRANSFORMATION_COMPLETE,
            EventType.OUTPUT_COMPLETE,
            EventType.WORKFLOW_STATE_CHANGED
        ]

    def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        self.agents[AgentRole.CEO] = CEOAgent()
        self.agents[AgentRole.CTO] = CTOAgent()
        self.agents[AgentRole.VC] = VCAgent()

        # Additional agents would be initialized here
        # self.agents[AgentRole.ANGEL] = AngelAgent()
        # self.agents[AgentRole.GROWTH] = GrowthAgent()
        # self.agents[AgentRole.VP_RD] = VPRDAgent()

        self.logger.info(f"Initialized {len(self.agents)} specialized agents")

    def _initialize_state_machine(self) -> list[StateTransition]:
        """Initialize the LangGraph state machine transitions."""
        return [
            StateTransition(
                from_state=WorkflowState.IDEATE,
                to_state=WorkflowState.VALIDATE,
                required_agents=[AgentRole.CEO, AgentRole.CTO],
                conditions={'confidence_threshold': 0.7}
            ),
            StateTransition(
                from_state=WorkflowState.VALIDATE,
                to_state=WorkflowState.RESEARCH,
                required_agents=[AgentRole.CEO, AgentRole.CTO],
                conditions={'technical_feasibility': True, 'market_feasibility': True}
            ),
            StateTransition(
                from_state=WorkflowState.RESEARCH,
                to_state=WorkflowState.DECK_GENERATION,
                required_agents=[AgentRole.VC],
                conditions={'evidence_count': 5}
            ),
            StateTransition(
                from_state=WorkflowState.DECK_GENERATION,
                to_state=WorkflowState.INVESTOR_EVALUATION,
                required_agents=[AgentRole.VC, AgentRole.CEO],
                conditions={'deck_quality': 0.8}
            ),
            StateTransition(
                from_state=WorkflowState.INVESTOR_EVALUATION,
                to_state=WorkflowState.SMOKE_TEST,
                required_agents=[AgentRole.CEO],
                conditions={'investment_readiness': True}
            ),
            StateTransition(
                from_state=WorkflowState.SMOKE_TEST,
                to_state=WorkflowState.MVP_GENERATION,
                required_agents=[AgentRole.CTO],
                conditions={'market_validation': True}
            ),
            StateTransition(
                from_state=WorkflowState.MVP_GENERATION,
                to_state=WorkflowState.DEPLOYMENT,
                required_agents=[AgentRole.CTO],
                conditions={'mvp_quality': 0.8}
            ),
            StateTransition(
                from_state=WorkflowState.DEPLOYMENT,
                to_state=WorkflowState.COMPLETED,
                required_agents=[AgentRole.CTO],
                conditions={'deployment_success': True}
            )
        ]

    async def handle(self, event: DomainEvent) -> None:
        """Handle domain events and trigger workflow transitions."""
        try:
            self.logger.info(
                f"Orchestrator handling event: {event.event_type.value}",
                extra={
                    'event_id': event.event_id,
                    'aggregate_id': event.aggregate_id,
                    'correlation_id': event.correlation_id
                }
            )

            if event.event_type == EventType.IDEA_CREATED:
                await self._start_workflow(event)
            elif event.event_type == EventType.WORKFLOW_STATE_CHANGED:
                await self._handle_state_change(event)
            else:
                await self._process_workflow_event(event)

        except Exception as e:
            self.logger.error(
                f"Error handling event {event.event_type.value}: {e}",
                extra={
                    'event_id': event.event_id,
                    'aggregate_id': event.aggregate_id,
                    'error': str(e)
                }
            )

    async def _start_workflow(self, event: DomainEvent) -> None:
        """Start a new workflow for a startup idea."""
        aggregate_id = event.aggregate_id
        startup_idea = event.event_data.get('idea', '')

        if not startup_idea:
            self.logger.error(f"No startup idea found in event {event.event_id}")
            return

        # Create workflow context
        context = AgentContext(
            current_state=WorkflowState.IDEATE,
            startup_idea=startup_idea,
            aggregate_id=aggregate_id,
            correlation_id=event.correlation_id or event.event_id,
            execution_id=f"workflow_{aggregate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            budget_remaining=60.0  # Total budget from settings
        )

        # Store active workflow
        self.active_workflows[aggregate_id] = context
        self.execution_metrics['total_workflows'] += 1

        self.logger.info(
            f"Started workflow for idea: {startup_idea[:100]}...",
            extra={
                'aggregate_id': aggregate_id,
                'execution_id': context.execution_id,
                'initial_state': context.current_state.value
            }
        )

        # Execute initial state
        await self._execute_workflow_state(context)

    async def _execute_workflow_state(self, context: AgentContext) -> None:
        """Execute the current workflow state with appropriate agents."""
        try:
            # Get agents that can handle current state
            available_agents = [
                agent for agent in self.agents.values()
                if agent.can_handle_state(context.current_state) and agent.active
            ]

            if not available_agents:
                self.logger.warning(
                    f"No agents available for state {context.current_state.value}",
                    extra={'aggregate_id': context.aggregate_id}
                )
                await self._transition_to_failed_state(context, "No available agents")
                return

            # Execute agents in parallel or sequence based on state requirements
            agent_decisions = []
            for agent in available_agents:
                try:
                    decision = await agent.execute(context)
                    agent_decisions.append(decision)

                    self.logger.info(
                        f"Agent {agent.role.value} completed execution",
                        extra={
                            'aggregate_id': context.aggregate_id,
                            'decision_type': decision.decision_type,
                            'confidence': decision.confidence_score,
                            'cost': decision.cost_estimate
                        }
                    )

                except Exception as e:
                    self.logger.error(
                        f"Agent {agent.role.value} execution failed: {e}",
                        extra={'aggregate_id': context.aggregate_id}
                    )

            # Process agent decisions and determine next state
            if agent_decisions:
                await self._process_agent_decisions(context, agent_decisions)
            else:
                await self._transition_to_failed_state(context, "All agent executions failed")

        except Exception as e:
            self.logger.error(
                f"Workflow state execution failed: {e}",
                extra={'aggregate_id': context.aggregate_id}
            )
            await self._transition_to_failed_state(context, str(e))

    async def _process_agent_decisions(
        self,
        context: AgentContext,
        decisions: list[AgentDecision]
    ) -> None:
        """Process agent decisions and determine workflow progression."""

        # Aggregate decisions and determine consensus
        consensus_next_state = self._determine_consensus_next_state(decisions)
        confidence_scores = [d.confidence_score for d in decisions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Update context with decision data
        for decision in decisions:
            context.state_data[f"{decision.agent_role.value}_decision"] = decision.to_dict()

        # Check transition conditions
        if consensus_next_state and avg_confidence >= 0.7:
            await self._transition_to_state(context, consensus_next_state)
        else:
            # If no consensus or low confidence, retry current state or fail
            retry_count = context.state_data.get('retry_count', 0)
            if retry_count < 2:
                context.state_data['retry_count'] = retry_count + 1
                self.logger.info(
                    f"Retrying state {context.current_state.value} (attempt {retry_count + 1})",
                    extra={'aggregate_id': context.aggregate_id}
                )
                await self._execute_workflow_state(context)
            else:
                await self._transition_to_failed_state(
                    context,
                    f"Failed to reach consensus after retries (confidence: {avg_confidence:.2f})"
                )

    def _determine_consensus_next_state(self, decisions: list[AgentDecision]) -> WorkflowState | None:
        """Determine consensus next state from agent decisions."""
        next_states = [d.next_state for d in decisions if d.next_state]

        if not next_states:
            return None

        # Simple majority vote
        state_counts = {}
        for state in next_states:
            state_counts[state] = state_counts.get(state, 0) + 1

        return max(state_counts.items(), key=lambda x: x[1])[0]

    async def _transition_to_state(self, context: AgentContext, new_state: WorkflowState) -> None:
        """Transition workflow to a new state."""
        old_state = context.current_state
        context.current_state = new_state
        context.timestamp = datetime.now(UTC)

        # Publish state change event
        await self.event_bus.publish(DomainEvent(
            event_type=EventType.WORKFLOW_STATE_CHANGED,
            aggregate_id=context.aggregate_id,
            correlation_id=context.correlation_id,
            event_data={
                'old_state': old_state.value,
                'new_state': new_state.value,
                'execution_id': context.execution_id,
                'timestamp': context.timestamp.isoformat()
            }
        ))

        self.logger.info(
            f"Workflow transitioned: {old_state.value} -> {new_state.value}",
            extra={
                'aggregate_id': context.aggregate_id,
                'execution_id': context.execution_id
            }
        )

        # Execute new state if not completed
        if new_state not in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            await self._execute_workflow_state(context)
        else:
            await self._complete_workflow(context)

    async def _transition_to_failed_state(self, context: AgentContext, reason: str) -> None:
        """Transition workflow to failed state."""
        context.current_state = WorkflowState.FAILED
        context.state_data['failure_reason'] = reason

        self.execution_metrics['failed_workflows'] += 1

        # Publish failure event
        await self.event_bus.publish(DomainEvent(
            event_type=EventType.PIPELINE_FAILED,
            aggregate_id=context.aggregate_id,
            correlation_id=context.correlation_id,
            event_data={
                'execution_id': context.execution_id,
                'failure_reason': reason,
                'final_state': context.current_state.value
            }
        ))

        self.logger.error(
            f"Workflow failed: {reason}",
            extra={
                'aggregate_id': context.aggregate_id,
                'execution_id': context.execution_id
            }
        )

        # Clean up
        if context.aggregate_id in self.active_workflows:
            del self.active_workflows[context.aggregate_id]

    async def _complete_workflow(self, context: AgentContext) -> None:
        """Complete workflow execution."""
        self.execution_metrics['completed_workflows'] += 1

        # Calculate execution metrics
        execution_time = (datetime.now(UTC) - context.timestamp).total_seconds()

        # Publish completion event
        await self.event_bus.publish(DomainEvent(
            event_type=EventType.OUTPUT_COMPLETE,
            aggregate_id=context.aggregate_id,
            correlation_id=context.correlation_id,
            event_data={
                'execution_id': context.execution_id,
                'execution_time': execution_time,
                'final_state': context.current_state.value,
                'agent_decisions': context.state_data
            }
        ))

        self.logger.info(
            "Workflow completed successfully",
            extra={
                'aggregate_id': context.aggregate_id,
                'execution_id': context.execution_id,
                'execution_time': execution_time
            }
        )

        # Clean up
        if context.aggregate_id in self.active_workflows:
            del self.active_workflows[context.aggregate_id]

    async def _handle_state_change(self, event: DomainEvent) -> None:
        """Handle workflow state change events."""
        aggregate_id = event.aggregate_id
        if aggregate_id not in self.active_workflows:
            self.logger.warning(f"Received state change for unknown workflow: {aggregate_id}")
            return

        context = self.active_workflows[aggregate_id]
        new_state_value = event.event_data.get('new_state')

        if new_state_value:
            try:
                new_state = WorkflowState(new_state_value)
                if context.current_state != new_state:
                    context.current_state = new_state
                    await self._execute_workflow_state(context)
            except ValueError:
                self.logger.error(f"Invalid workflow state: {new_state_value}")

    async def _process_workflow_event(self, event: DomainEvent) -> None:
        """Process workflow-related events."""
        aggregate_id = event.aggregate_id
        if aggregate_id not in self.active_workflows:
            return

        context = self.active_workflows[aggregate_id]

        # Update context with event data
        context.state_data[f"event_{event.event_type.value}"] = event.event_data

        # Check if event triggers state transition
        if event.event_type == EventType.PROCESSING_COMPLETE:
            if context.current_state == WorkflowState.RESEARCH:
                await self._transition_to_state(context, WorkflowState.DECK_GENERATION)

        elif event.event_type == EventType.TRANSFORMATION_COMPLETE:
            if context.current_state == WorkflowState.DECK_GENERATION:
                await self._transition_to_state(context, WorkflowState.INVESTOR_EVALUATION)

    async def get_workflow_status(self, aggregate_id: str) -> dict[str, Any] | None:
        """Get current workflow status."""
        if aggregate_id not in self.active_workflows:
            return None

        context = self.active_workflows[aggregate_id]
        return {
            'aggregate_id': aggregate_id,
            'execution_id': context.execution_id,
            'current_state': context.current_state.value,
            'startup_idea': context.startup_idea,
            'budget_remaining': context.budget_remaining,
            'agent_messages': len(context.agent_messages),
            'state_data_keys': list(context.state_data.keys()),
            'timestamp': context.timestamp.isoformat()
        }

    async def get_orchestrator_metrics(self) -> dict[str, Any]:
        """Get orchestrator performance metrics."""
        return {
            **self.execution_metrics,
            'active_workflows': len(self.active_workflows),
            'registered_agents': len(self.agents),
            'state_transitions': len(self.state_transitions)
        }


# Singleton instance
_agent_orchestrator: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get singleton Agent Orchestrator instance."""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        _agent_orchestrator = AgentOrchestrator()
    return _agent_orchestrator
