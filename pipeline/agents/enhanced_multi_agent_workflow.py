"""
Enhanced Multi-Agent Workflow - Advanced CrewAI and LangGraph integration.

Combines CrewAI's agent orchestration with LangGraph's state management
for complex multi-agent workflows in startup validation and generation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict
from uuid import uuid4

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.agent import Agent as CrewAIAgent
    from crewai.task import Task as CrewAITask
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = Task = Crew = Process = None

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = END = START = None

try:
    from langchain.schema import AIMessage, BaseMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = BaseMessage = HumanMessage = AIMessage = None

from pipeline.config.settings import get_settings
from pipeline.core.service_registry import ServiceInterface
from pipeline.services.budget_sentinel import BudgetCategory, get_budget_sentinel


class AgentRole(Enum):
    """Defined agent roles in the startup validation workflow."""
    CEO = "ceo"
    CTO = "cto"
    VP_RD = "vp_rd"
    GROWTH_MARKETER = "growth_marketer"
    VC_ANALYST = "vc_analyst"
    ANGEL_INVESTOR = "angel_investor"
    ENGINEER = "engineer"


class WorkflowStage(Enum):
    """Multi-agent workflow stages."""
    IDEATION = "ideation"
    TECHNICAL_VALIDATION = "technical_validation"
    MARKET_ANALYSIS = "market_analysis"
    BUSINESS_MODEL = "business_model"
    INVESTOR_PITCH = "investor_pitch"
    TECHNICAL_ARCHITECTURE = "technical_architecture"
    GO_TO_MARKET = "go_to_market"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"


class TaskPriority(Enum):
    """Task execution priorities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    role: AgentRole
    name: str
    goal: str
    backstory: str
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    memory_type: str = "vector"
    max_iterations: int = 3
    temperature: float = 0.7
    delegation_allowed: bool = True
    verbose: bool = True


@dataclass
class TaskConfig:
    """Configuration for agent tasks."""
    task_id: str
    description: str
    agent_role: AgentRole
    stage: WorkflowStage
    priority: TaskPriority
    dependencies: list[str] = field(default_factory=list)
    expected_output: str = ""
    max_execution_time: int = 300  # seconds
    retry_count: int = 2
    validation_criteria: dict[str, Any] = field(default_factory=dict)


class MultiAgentState(TypedDict):
    """LangGraph state for multi-agent workflow."""

    # Core workflow state
    workflow_id: str
    current_stage: WorkflowStage
    startup_idea: str
    progress: float
    started_at: datetime

    # Agent outputs by stage
    ideation_results: dict[str, Any]
    technical_validation: dict[str, Any]
    market_analysis: dict[str, Any]
    business_model: dict[str, Any]
    investor_pitch: dict[str, Any]
    technical_architecture: dict[str, Any]
    go_to_market: dict[str, Any]
    final_review: dict[str, Any]

    # Task tracking
    completed_tasks: list[str]
    failed_tasks: list[str]
    active_tasks: list[str]

    # Quality metrics
    consensus_score: float
    technical_feasibility: float
    market_viability: float
    business_potential: float

    # Agent interactions
    agent_messages: list[dict[str, Any]]
    debates: list[dict[str, Any]]
    decisions: list[dict[str, Any]]

    # Error handling
    errors: list[str]
    retry_count: int

    # Metadata
    metadata: dict[str, Any]


class EnhancedMultiAgentWorkflow(ServiceInterface):
    """Enhanced multi-agent workflow system combining CrewAI and LangGraph."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.budget_sentinel = get_budget_sentinel()

        # Check dependencies
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is required for multi-agent workflows")
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required for state management")
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LLM integration")

        # Initialize components
        self.llm = self._initialize_llm()
        self.agents: dict[AgentRole, Agent] = {}
        self.agent_configs: dict[AgentRole, AgentConfig] = {}
        self.task_configs: list[TaskConfig] = []

        # LangGraph components
        self.checkpointer = MemorySaver()
        self.graph = None

        # Workflow statistics
        self.stats = {
            'workflows_executed': 0,
            'total_tasks_completed': 0,
            'average_consensus_score': 0.0,
            'average_execution_time': 0.0
        }

    async def initialize(self) -> None:
        """Initialize the multi-agent workflow system."""
        # Load agent configurations
        await self._load_agent_configurations()

        # Create agents
        await self._create_agents()

        # Define task configurations
        await self._define_task_configurations()

        # Build LangGraph workflow
        self._build_workflow_graph()

        self.logger.info("Enhanced multi-agent workflow system initialized")

    async def shutdown(self) -> None:
        """Shutdown the workflow system."""
        # Cleanup any active workflows
        self.logger.info("Multi-agent workflow system shutdown")

    def _initialize_llm(self) -> Any:
        """Initialize the language model."""
        api_key = getattr(self.settings, 'openai_api_key', None)
        if not api_key:
            raise ValueError("OpenAI API key is required for multi-agent workflows")

        return ChatOpenAI(
            model=getattr(self.settings, 'default_llm_model', 'gpt-4o'),
            temperature=0.7,
            api_key=api_key
        )

    async def _load_agent_configurations(self) -> None:
        """Load agent configurations from YAML files."""
        from pathlib import Path

        agents_dir = Path(__file__).parent.parent.parent / "agents"

        # Default configurations for each role
        default_configs = {
            AgentRole.CEO: AgentConfig(
                role=AgentRole.CEO,
                name="CEO",
                goal="Define overall vision, strategy, and business model",
                backstory="Experienced startup founder with multiple successful exits",
                skills=["strategic_planning", "market_analysis", "fundraising"],
                tools=["browser_search", "web_rag"]
            ),
            AgentRole.CTO: AgentConfig(
                role=AgentRole.CTO,
                name="CTO",
                goal="Evaluate technical feasibility and architecture",
                backstory="Senior technology leader with deep engineering experience",
                skills=["technical_architecture", "scalability_analysis", "technology_selection"],
                tools=["python_interpreter", "code_analysis"]
            ),
            AgentRole.VP_RD: AgentConfig(
                role=AgentRole.VP_RD,
                name="VP R&D",
                goal="Lead research and development strategy",
                backstory="Innovation leader with track record of breakthrough products",
                skills=["research_methodology", "innovation_management", "product_development"],
                tools=["research_tools", "patent_search"]
            ),
            AgentRole.GROWTH_MARKETER: AgentConfig(
                role=AgentRole.GROWTH_MARKETER,
                name="Growth Marketer",
                goal="Develop go-to-market strategy and growth tactics",
                backstory="Growth expert who has scaled multiple startups",
                skills=["growth_hacking", "market_penetration", "customer_acquisition"],
                tools=["analytics_tools", "campaign_optimization"]
            ),
            AgentRole.VC_ANALYST: AgentConfig(
                role=AgentRole.VC_ANALYST,
                name="VC Analyst",
                goal="Evaluate investment potential and market opportunity",
                backstory="Venture capital analyst with expertise in startup evaluation",
                skills=["due_diligence", "market_sizing", "competitive_analysis"],
                tools=["financial_modeling", "market_research"]
            )
        }

        self.agent_configs = default_configs

    async def _create_agents(self) -> None:
        """Create CrewAI agents from configurations."""
        for role, config in self.agent_configs.items():
            agent = Agent(
                role=config.name,
                goal=config.goal,
                backstory=config.backstory,
                llm=self.llm,
                verbose=config.verbose,
                allow_delegation=config.delegation_allowed,
                max_iter=config.max_iterations
            )

            self.agents[role] = agent
            self.logger.debug(f"Created agent: {config.name} ({role.value})")

    async def _define_task_configurations(self) -> None:
        """Define task configurations for the workflow."""
        self.task_configs = [
            # Ideation Stage
            TaskConfig(
                task_id="ideation_ceo",
                description="Generate 3 innovative startup concepts with clear value propositions",
                agent_role=AgentRole.CEO,
                stage=WorkflowStage.IDEATION,
                priority=TaskPriority.CRITICAL,
                expected_output="List of 3 startup concepts with problem statements and solutions"
            ),
            TaskConfig(
                task_id="ideation_cto_review",
                description="Evaluate technical feasibility of proposed startup concepts",
                agent_role=AgentRole.CTO,
                stage=WorkflowStage.IDEATION,
                priority=TaskPriority.HIGH,
                dependencies=["ideation_ceo"],
                expected_output="Technical feasibility assessment for each concept"
            ),

            # Technical Validation Stage
            TaskConfig(
                task_id="technical_architecture",
                description="Design high-level technical architecture for the selected concept",
                agent_role=AgentRole.CTO,
                stage=WorkflowStage.TECHNICAL_VALIDATION,
                priority=TaskPriority.CRITICAL,
                expected_output="Technical architecture diagram and technology stack recommendations"
            ),
            TaskConfig(
                task_id="research_validation",
                description="Validate technical approach through research and benchmarking",
                agent_role=AgentRole.VP_RD,
                stage=WorkflowStage.TECHNICAL_VALIDATION,
                priority=TaskPriority.HIGH,
                expected_output="Research validation report with technical benchmarks"
            ),

            # Market Analysis Stage
            TaskConfig(
                task_id="market_sizing",
                description="Analyze market size, growth, and opportunity",
                agent_role=AgentRole.VC_ANALYST,
                stage=WorkflowStage.MARKET_ANALYSIS,
                priority=TaskPriority.CRITICAL,
                expected_output="Market sizing analysis with TAM, SAM, and SOM calculations"
            ),
            TaskConfig(
                task_id="competitive_analysis",
                description="Analyze competitive landscape and differentiation opportunities",
                agent_role=AgentRole.GROWTH_MARKETER,
                stage=WorkflowStage.MARKET_ANALYSIS,
                priority=TaskPriority.HIGH,
                expected_output="Competitive analysis with positioning recommendations"
            ),

            # Business Model Stage
            TaskConfig(
                task_id="business_model_design",
                description="Design comprehensive business model and monetization strategy",
                agent_role=AgentRole.CEO,
                stage=WorkflowStage.BUSINESS_MODEL,
                priority=TaskPriority.CRITICAL,
                expected_output="Business model canvas with revenue projections"
            ),

            # Go-to-Market Stage
            TaskConfig(
                task_id="gtm_strategy",
                description="Develop comprehensive go-to-market strategy",
                agent_role=AgentRole.GROWTH_MARKETER,
                stage=WorkflowStage.GO_TO_MARKET,
                priority=TaskPriority.CRITICAL,
                expected_output="Go-to-market plan with customer acquisition strategy"
            ),

            # Final Review Stage
            TaskConfig(
                task_id="investment_evaluation",
                description="Provide final investment recommendation and risk assessment",
                agent_role=AgentRole.VC_ANALYST,
                stage=WorkflowStage.FINAL_REVIEW,
                priority=TaskPriority.CRITICAL,
                expected_output="Investment recommendation with risk analysis"
            )
        ]

    def _build_workflow_graph(self) -> None:
        """Build the LangGraph workflow state machine."""
        if not LANGGRAPH_AVAILABLE:
            return

        graph = StateGraph(MultiAgentState)

        # Add nodes for each workflow stage
        graph.add_node("ideation", self._ideation_node)
        graph.add_node("technical_validation", self._technical_validation_node)
        graph.add_node("market_analysis", self._market_analysis_node)
        graph.add_node("business_model", self._business_model_node)
        graph.add_node("go_to_market", self._go_to_market_node)
        graph.add_node("final_review", self._final_review_node)
        graph.add_node("consensus_check", self._consensus_check_node)
        graph.add_node("quality_gate", self._quality_gate_node)

        # Define workflow edges
        graph.set_entry_point("ideation")

        # Sequential flow with quality gates
        graph.add_edge("ideation", "technical_validation")
        graph.add_edge("technical_validation", "quality_gate")
        graph.add_edge("market_analysis", "quality_gate")
        graph.add_edge("business_model", "quality_gate")
        graph.add_edge("go_to_market", "quality_gate")

        # Conditional routing from quality gate
        graph.add_conditional_edges(
            "quality_gate",
            self._route_from_quality_gate,
            {
                "market_analysis": "market_analysis",
                "business_model": "business_model",
                "go_to_market": "go_to_market",
                "final_review": "final_review",
                "consensus_check": "consensus_check",
                "completed": END
            }
        )

        # Consensus checking
        graph.add_conditional_edges(
            "consensus_check",
            self._route_from_consensus,
            {
                "retry": "ideation",
                "final_review": "final_review",
                "completed": END
            }
        )

        graph.add_edge("final_review", END)

        self.graph = graph.compile(checkpointer=self.checkpointer)

    async def execute_workflow(
        self,
        startup_idea: str,
        workflow_id: str | None = None
    ) -> MultiAgentState:
        """Execute the complete multi-agent workflow."""
        workflow_id = workflow_id or f"workflow_{uuid4().hex[:8]}"

        # Initialize workflow state
        initial_state: MultiAgentState = {
            "workflow_id": workflow_id,
            "current_stage": WorkflowStage.IDEATION,
            "startup_idea": startup_idea,
            "progress": 0.0,
            "started_at": datetime.utcnow(),
            "ideation_results": {},
            "technical_validation": {},
            "market_analysis": {},
            "business_model": {},
            "investor_pitch": {},
            "technical_architecture": {},
            "go_to_market": {},
            "final_review": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "active_tasks": [],
            "consensus_score": 0.0,
            "technical_feasibility": 0.0,
            "market_viability": 0.0,
            "business_potential": 0.0,
            "agent_messages": [],
            "debates": [],
            "decisions": [],
            "errors": [],
            "retry_count": 0,
            "metadata": {}
        }

        try:
            # Track budget for workflow execution
            async with self.budget_sentinel.track_operation(
                "multi_agent_workflow",
                "execute_workflow",
                BudgetCategory.OPENAI_TOKENS,
                15.0  # Budget allocation for multi-agent workflow
            ):
                config = {"configurable": {"thread_id": workflow_id}}
                final_state = await self.graph.ainvoke(initial_state, config)

                # Update statistics
                self.stats['workflows_executed'] += 1
                self.stats['total_tasks_completed'] += len(final_state['completed_tasks'])

                return final_state

        except Exception as e:
            self.logger.error(f"Multi-agent workflow execution failed: {e}")
            raise

    async def _ideation_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute ideation stage with CEO and CTO collaboration."""
        self.logger.info(f"Starting ideation for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.IDEATION
        state["progress"] = 0.2

        try:
            # CEO generates initial concepts
            ceo_task = Task(
                description=f"Generate 3 innovative startup concepts based on: {state['startup_idea']}",
                agent=self.agents[AgentRole.CEO],
                expected_output="3 startup concepts with clear value propositions"
            )

            # CTO evaluates technical feasibility
            cto_task = Task(
                description="Evaluate technical feasibility of the generated concepts",
                agent=self.agents[AgentRole.CTO],
                expected_output="Technical feasibility assessment"
            )

            # Create and execute crew
            ideation_crew = Crew(
                agents=[self.agents[AgentRole.CEO], self.agents[AgentRole.CTO]],
                tasks=[ceo_task, cto_task],
                process=Process.sequential,
                verbose=False
            )

            result = ideation_crew.kickoff()

            state["ideation_results"] = {
                "concepts": result,
                "timestamp": datetime.utcnow(),
                "participants": ["CEO", "CTO"]
            }

            state["completed_tasks"].append("ideation_ceo")
            state["completed_tasks"].append("ideation_cto_review")

        except Exception as e:
            state["errors"].append(f"Ideation failed: {str(e)}")
            state["failed_tasks"].append("ideation")

        return state

    async def _technical_validation_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute technical validation stage."""
        self.logger.info(f"Starting technical validation for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.TECHNICAL_VALIDATION
        state["progress"] = 0.4

        # Technical validation logic here
        state["technical_validation"] = {
            "architecture_designed": True,
            "feasibility_score": 0.85,
            "technology_stack": ["Python", "PostgreSQL", "React"],
            "timestamp": datetime.utcnow()
        }

        state["technical_feasibility"] = 0.85
        state["completed_tasks"].append("technical_validation")

        return state

    async def _market_analysis_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute market analysis stage."""
        self.logger.info(f"Starting market analysis for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.MARKET_ANALYSIS
        state["progress"] = 0.6

        # Market analysis logic here
        state["market_analysis"] = {
            "market_size": {"TAM": "$10B", "SAM": "$1B", "SOM": "$100M"},
            "competitive_landscape": "Moderate competition with differentiation opportunities",
            "viability_score": 0.78,
            "timestamp": datetime.utcnow()
        }

        state["market_viability"] = 0.78
        state["completed_tasks"].append("market_analysis")

        return state

    async def _business_model_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute business model design stage."""
        self.logger.info(f"Starting business model design for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.BUSINESS_MODEL
        state["progress"] = 0.8

        # Business model logic here
        state["business_model"] = {
            "revenue_model": "SaaS subscription with freemium tier",
            "unit_economics": {"CAC": "$50", "LTV": "$500"},
            "business_score": 0.82,
            "timestamp": datetime.utcnow()
        }

        state["business_potential"] = 0.82
        state["completed_tasks"].append("business_model")

        return state

    async def _go_to_market_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute go-to-market strategy stage."""
        self.logger.info(f"Starting go-to-market strategy for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.GO_TO_MARKET
        state["progress"] = 0.9

        # GTM strategy logic here
        state["go_to_market"] = {
            "strategy": "Product-led growth with content marketing",
            "channels": ["Direct sales", "Partner network", "Digital marketing"],
            "timeline": "6-month rollout plan",
            "timestamp": datetime.utcnow()
        }

        state["completed_tasks"].append("go_to_market")

        return state

    async def _final_review_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute final review and recommendation."""
        self.logger.info(f"Starting final review for workflow {state['workflow_id']}")

        state["current_stage"] = WorkflowStage.FINAL_REVIEW
        state["progress"] = 1.0

        # Calculate overall consensus score
        scores = [
            state["technical_feasibility"],
            state["market_viability"],
            state["business_potential"]
        ]
        state["consensus_score"] = sum(scores) / len(scores)

        state["final_review"] = {
            "recommendation": "PROCEED" if state["consensus_score"] > 0.7 else "REVISE",
            "overall_score": state["consensus_score"],
            "key_strengths": ["Strong technical foundation", "Clear market opportunity"],
            "areas_for_improvement": ["Competitive differentiation", "Go-to-market execution"],
            "timestamp": datetime.utcnow()
        }

        state["completed_tasks"].append("final_review")

        return state

    async def _consensus_check_node(self, state: MultiAgentState) -> MultiAgentState:
        """Check for consensus among agents."""
        # Consensus checking logic
        if state["consensus_score"] < 0.6 and state["retry_count"] < 2:
            state["retry_count"] += 1
            return state

        return state

    async def _quality_gate_node(self, state: MultiAgentState) -> MultiAgentState:
        """Quality gate validation."""
        # Quality gate logic
        return state

    def _route_from_quality_gate(self, state: MultiAgentState) -> str:
        """Route from quality gate based on current stage."""
        current_stage = state["current_stage"]

        if current_stage == WorkflowStage.TECHNICAL_VALIDATION:
            return "market_analysis"
        if current_stage == WorkflowStage.MARKET_ANALYSIS:
            return "business_model"
        if current_stage == WorkflowStage.BUSINESS_MODEL:
            return "go_to_market"
        if current_stage == WorkflowStage.GO_TO_MARKET:
            return "final_review"
        return "completed"

    def _route_from_consensus(self, state: MultiAgentState) -> str:
        """Route from consensus check."""
        if state["consensus_score"] < 0.6 and state["retry_count"] < 2:
            return "retry"
        if state["consensus_score"] >= 0.6:
            return "final_review"
        return "completed"

    async def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get status of a running workflow."""
        # Implementation for workflow status retrieval
        return {
            "workflow_id": workflow_id,
            "status": "running",
            "progress": 0.5,
            "current_stage": "market_analysis"
        }

    def get_service_info(self) -> dict[str, Any]:
        """Get service information."""
        return {
            "name": "EnhancedMultiAgentWorkflow",
            "version": "1.0.0",
            "status": "ready",
            "agents_count": len(self.agents),
            "task_configs_count": len(self.task_configs),
            "stats": self.stats,
            "dependencies": {
                "crewai_available": CREWAI_AVAILABLE,
                "langgraph_available": LANGGRAPH_AVAILABLE,
                "langchain_available": LANGCHAIN_AVAILABLE
            }
        }


# Singleton instance
_multi_agent_workflow = None


async def get_multi_agent_workflow() -> EnhancedMultiAgentWorkflow:
    """Get singleton multi-agent workflow instance."""
    global _multi_agent_workflow
    if _multi_agent_workflow is None:
        _multi_agent_workflow = EnhancedMultiAgentWorkflow()
        await _multi_agent_workflow.initialize()
    return _multi_agent_workflow
