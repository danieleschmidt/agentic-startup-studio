"""
Workflow Orchestrator Service - LangGraph-based state machine for pipeline orchestration.

Coordinates the end-to-end startup idea validation workflow across all 4 phases:
1. Data Ingestion - Idea capture and validation
2. Data Processing - RAG evidence collection and quality scoring  
3. Data Transformation - Pitch deck generation and investor evaluation
4. Data Output - Smoke test campaigns and MVP deployment
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, TypedDict, Callable
from dataclasses import dataclass, field
import asyncio
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pipeline.config.settings import get_settings
from pipeline.services.budget_sentinel import get_budget_sentinel, BudgetCategory, BudgetExceededException


class PipelineStage(Enum):
    """Pipeline stage definitions."""
    IDEATE = "ideate"
    RESEARCH = "research" 
    DECK_GENERATION = "deck_generation"
    INVESTOR_EVALUATION = "investor_evaluation"
    SMOKE_TEST = "smoke_test"
    MVP_GENERATION = "mvp_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityGateResult(Enum):
    """Quality gate validation results."""
    PASSED = "passed"
    FAILED = "failed"
    BYPASSED = "bypassed"


@dataclass
class WorkflowCheckpoint:
    """Workflow checkpoint for resume capability."""
    stage: PipelineStage
    timestamp: datetime
    data: Dict[str, Any]
    progress: float  # 0.0 to 1.0


class WorkflowState(TypedDict):
    """LangGraph state definition for workflow coordination."""
    
    # Core workflow state
    idea_id: str
    current_stage: PipelineStage
    progress: float
    started_at: datetime
    
    # Stage-specific data
    idea_data: Dict[str, Any]
    research_data: Dict[str, Any]
    deck_data: Dict[str, Any]
    investor_data: Dict[str, Any]
    smoke_test_data: Dict[str, Any]
    mvp_data: Dict[str, Any]
    
    # Quality gates
    quality_gates: Dict[str, QualityGateResult]
    
    # Error handling
    errors: List[str]
    retry_count: int
    
    # Budget tracking
    costs_tracked: Dict[str, float]
    
    # Metadata
    metadata: Dict[str, Any]


class WorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for pipeline coordination."""
    
    def __init__(self):
        self.settings = get_settings()
        self.budget_sentinel = get_budget_sentinel()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangGraph components
        self.checkpointer = MemorySaver()
        self.graph = self._build_workflow_graph()
        
        # Stage processors registry
        self.stage_processors: Dict[PipelineStage, Callable] = {
            PipelineStage.IDEATE: self._process_ideate_stage,
            PipelineStage.RESEARCH: self._process_research_stage,
            PipelineStage.DECK_GENERATION: self._process_deck_generation_stage,
            PipelineStage.INVESTOR_EVALUATION: self._process_investor_evaluation_stage,
            PipelineStage.SMOKE_TEST: self._process_smoke_test_stage,
            PipelineStage.MVP_GENERATION: self._process_mvp_generation_stage,
        }
        
        # Quality gate validators
        self.quality_gates: Dict[PipelineStage, Callable] = {
            PipelineStage.RESEARCH: self._validate_research_quality_gate,
            PipelineStage.DECK_GENERATION: self._validate_deck_quality_gate,
            PipelineStage.INVESTOR_EVALUATION: self._validate_investor_quality_gate,
        }
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow state machine."""
        graph = StateGraph(WorkflowState)
        
        # Add nodes for each stage
        graph.add_node("ideate", self._ideate_node)
        graph.add_node("research", self._research_node)
        graph.add_node("deck_generation", self._deck_generation_node)
        graph.add_node("investor_evaluation", self._investor_evaluation_node)
        graph.add_node("smoke_test", self._smoke_test_node)
        graph.add_node("mvp_generation", self._mvp_generation_node)
        graph.add_node("quality_gate", self._quality_gate_node)
        graph.add_node("error_handler", self._error_handler_node)
        
        # Define workflow edges
        graph.set_entry_point("ideate")
        
        # Linear progression with quality gates
        graph.add_edge("ideate", "research")
        graph.add_edge("research", "quality_gate")
        graph.add_edge("deck_generation", "quality_gate")
        graph.add_edge("investor_evaluation", "quality_gate")
        
        # Conditional routing from quality gate
        graph.add_conditional_edges(
            "quality_gate",
            self._route_from_quality_gate,
            {
                "deck_generation": "deck_generation",
                "investor_evaluation": "investor_evaluation", 
                "smoke_test": "smoke_test",
                "mvp_generation": "mvp_generation",
                "completed": END,
                "error": "error_handler",
                "retry": "research"  # Retry from research on failure
            }
        )
        
        # Error handling routes
        graph.add_conditional_edges(
            "error_handler",
            self._route_from_error_handler,
            {
                "retry": "research",
                "failed": END
            }
        )
        
        # Terminal nodes
        graph.add_edge("smoke_test", "mvp_generation")
        graph.add_edge("mvp_generation", END)
        
        return graph.compile(checkpointer=self.checkpointer)
    
    async def execute_workflow(self, idea_id: str, idea_data: Dict[str, Any]) -> WorkflowState:
        """Execute the complete workflow for an idea."""
        config = {"configurable": {"thread_id": idea_id}}
        
        # Initialize workflow state
        initial_state: WorkflowState = {
            "idea_id": idea_id,
            "current_stage": PipelineStage.IDEATE,
            "progress": 0.0,
            "started_at": datetime.utcnow(),
            "idea_data": idea_data,
            "research_data": {},
            "deck_data": {},
            "investor_data": {},
            "smoke_test_data": {},
            "mvp_data": {},
            "quality_gates": {},
            "errors": [],
            "retry_count": 0,
            "costs_tracked": {},
            "metadata": {}
        }
        
        try:
            # Execute workflow with budget tracking
            async with self.budget_sentinel.track_operation(
                "workflow_orchestrator",
                "execute_pipeline",
                BudgetCategory.INFRASTRUCTURE,
                self.settings.budget.infrastructure_budget * 0.1  # 10% allocation
            ):
                final_state = await self.graph.ainvoke(initial_state, config)
                return final_state
                
        except BudgetExceededException as e:
            self.logger.error(f"Workflow execution blocked: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def resume_workflow(self, idea_id: str) -> Optional[WorkflowState]:
        """Resume workflow from last checkpoint."""
        config = {"configurable": {"thread_id": idea_id}}
        
        try:
            # Get last checkpoint
            checkpoint = await self.graph.aget_state(config)
            if not checkpoint:
                self.logger.warning(f"No checkpoint found for idea {idea_id}")
                return None
            
            # Resume execution
            final_state = await self.graph.ainvoke(checkpoint.values, config)
            return final_state
            
        except Exception as e:
            self.logger.error(f"Failed to resume workflow for {idea_id}: {e}")
            raise
    
    async def _ideate_node(self, state: WorkflowState) -> WorkflowState:
        """Process ideation stage."""
        self.logger.info(f"Processing ideation for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.IDEATE
        state["progress"] = 0.1
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.IDEATE](state)
        
        return state
    
    async def _research_node(self, state: WorkflowState) -> WorkflowState:
        """Process research stage."""
        self.logger.info(f"Processing research for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.RESEARCH
        state["progress"] = 0.3
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.RESEARCH](state)
        
        return state
    
    async def _deck_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Process deck generation stage."""
        self.logger.info(f"Processing deck generation for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.DECK_GENERATION
        state["progress"] = 0.5
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.DECK_GENERATION](state)
        
        return state
    
    async def _investor_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Process investor evaluation stage."""
        self.logger.info(f"Processing investor evaluation for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.INVESTOR_EVALUATION
        state["progress"] = 0.7
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.INVESTOR_EVALUATION](state)
        
        return state
    
    async def _smoke_test_node(self, state: WorkflowState) -> WorkflowState:
        """Process smoke test stage."""
        self.logger.info(f"Processing smoke test for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.SMOKE_TEST
        state["progress"] = 0.9
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.SMOKE_TEST](state)
        
        return state
    
    async def _mvp_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Process MVP generation stage."""
        self.logger.info(f"Processing MVP generation for {state['idea_id']}")
        
        state["current_stage"] = PipelineStage.MVP_GENERATION
        state["progress"] = 1.0
        
        # Process through stage processor
        await self.stage_processors[PipelineStage.MVP_GENERATION](state)
        
        return state
    
    async def _quality_gate_node(self, state: WorkflowState) -> WorkflowState:
        """Process quality gate validation."""
        current_stage = state["current_stage"]
        
        if current_stage in self.quality_gates:
            try:
                result = await self.quality_gates[current_stage](state)
                state["quality_gates"][current_stage.value] = result
                
                self.logger.info(
                    f"Quality gate {current_stage.value} result: {result.value} "
                    f"for {state['idea_id']}"
                )
                
            except Exception as e:
                self.logger.error(f"Quality gate validation failed: {e}")
                state["quality_gates"][current_stage.value] = QualityGateResult.FAILED
                state["errors"].append(f"Quality gate error: {e}")
        
        return state
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors with retry logic."""
        state["retry_count"] += 1
        max_retries = 3
        
        if state["retry_count"] <= max_retries:
            self.logger.warning(
                f"Retrying workflow for {state['idea_id']} "
                f"(attempt {state['retry_count']}/{max_retries})"
            )
            
            # Exponential backoff
            await asyncio.sleep(2 ** state["retry_count"])
        else:
            self.logger.error(
                f"Workflow failed after {max_retries} retries for {state['idea_id']}"
            )
            state["current_stage"] = PipelineStage.FAILED
        
        return state
    
    def _route_from_quality_gate(self, state: WorkflowState) -> str:
        """Route based on quality gate results."""
        current_stage = state["current_stage"]
        gate_result = state["quality_gates"].get(current_stage.value)
        
        if gate_result == QualityGateResult.FAILED:
            return "error"
        elif gate_result == QualityGateResult.PASSED or gate_result == QualityGateResult.BYPASSED:
            # Route to next stage
            if current_stage == PipelineStage.RESEARCH:
                return "deck_generation"
            elif current_stage == PipelineStage.DECK_GENERATION:
                return "investor_evaluation"
            elif current_stage == PipelineStage.INVESTOR_EVALUATION:
                return "smoke_test"
        
        return "error"
    
    def _route_from_error_handler(self, state: WorkflowState) -> str:
        """Route based on error handler results."""
        if state["retry_count"] <= 3:
            return "retry"
        else:
            return "failed"
    
    # Placeholder stage processors (to be implemented)
    async def _process_ideate_stage(self, state: WorkflowState):
        """Process ideation stage - placeholder for actual implementation."""
        state["metadata"]["ideate_processed"] = True
        await asyncio.sleep(0.1)  # Simulate processing
    
    async def _process_research_stage(self, state: WorkflowState):
        """Process research stage - placeholder for actual implementation."""
        state["research_data"] = {"evidence_count": 3, "quality_score": 0.8}
        await asyncio.sleep(0.1)  # Simulate processing
    
    async def _process_deck_generation_stage(self, state: WorkflowState):
        """Process deck generation stage - placeholder for actual implementation."""
        state["deck_data"] = {"slide_count": 10, "accessibility_score": 95}
        await asyncio.sleep(0.1)  # Simulate processing
    
    async def _process_investor_evaluation_stage(self, state: WorkflowState):
        """Process investor evaluation stage - placeholder for actual implementation."""
        state["investor_data"] = {"investor_score": 0.85, "consensus": 0.8}
        await asyncio.sleep(0.1)  # Simulate processing
    
    async def _process_smoke_test_stage(self, state: WorkflowState):
        """Process smoke test stage - placeholder for actual implementation."""
        state["smoke_test_data"] = {"campaign_id": "camp_123", "budget": 50.0}
        await asyncio.sleep(0.1)  # Simulate processing
    
    async def _process_mvp_generation_stage(self, state: WorkflowState):
        """Process MVP generation stage - placeholder for actual implementation."""
        state["mvp_data"] = {"mvp_url": "https://mvp.example.com", "health_checks": True}
        await asyncio.sleep(0.1)  # Simulate processing
    
    # Quality gate validators
    async def _validate_research_quality_gate(self, state: WorkflowState) -> QualityGateResult:
        """Validate research quality gate."""
        research_data = state.get("research_data", {})
        evidence_count = research_data.get("evidence_count", 0)
        quality_score = research_data.get("quality_score", 0.0)
        
        if evidence_count >= 3 and quality_score >= 0.7:
            return QualityGateResult.PASSED
        else:
            return QualityGateResult.FAILED
    
    async def _validate_deck_quality_gate(self, state: WorkflowState) -> QualityGateResult:
        """Validate deck quality gate."""
        deck_data = state.get("deck_data", {})
        slide_count = deck_data.get("slide_count", 0)
        accessibility_score = deck_data.get("accessibility_score", 0)
        
        if slide_count == 10 and accessibility_score >= 90:
            return QualityGateResult.PASSED
        else:
            return QualityGateResult.FAILED
    
    async def _validate_investor_quality_gate(self, state: WorkflowState) -> QualityGateResult:
        """Validate investor quality gate."""
        investor_data = state.get("investor_data", {})
        investor_score = investor_data.get("investor_score", 0.0)
        
        funding_threshold = self.settings.budget.total_cycle_budget  # Use as placeholder
        
        if investor_score >= 0.8:  # Configurable threshold
            return QualityGateResult.PASSED
        else:
            return QualityGateResult.FAILED


# Singleton instance
_workflow_orchestrator = None


def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get singleton Workflow Orchestrator instance."""
    global _workflow_orchestrator
    if _workflow_orchestrator is None:
        _workflow_orchestrator = WorkflowOrchestrator()
    return _workflow_orchestrator