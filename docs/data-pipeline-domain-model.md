# Data Pipeline Domain Model - Agentic Startup Studio

## 1. Domain Overview

### 1.1 Bounded Context
The Agentic Startup Studio operates within the **Startup Validation Domain**, encompassing idea generation, research validation, investor evaluation, and market testing through automated workflows.

### 1.2 Domain Glossary

| Term | Definition |
|------|------------|
| **Idea** | A structured startup concept with description, market analysis, and validation status |
| **Evidence** | Research citations, market data, and supporting documentation for idea validation |
| **Agent** | Autonomous AI entity responsible for specific workflow tasks (CEO, CTO, Investor, etc.) |
| **Rubric** | Weighted scoring criteria used by investor agents for idea evaluation |
| **Smoke Test** | Market validation campaign using landing pages and paid advertising |
| **Pipeline State** | Current stage in the workflow: Ideate→Research→Deck→Investors→Smoke Test |
| **Budget Sentinel** | Cost monitoring system with automatic spending controls |
| **Quality Gate** | Automated validation checkpoint ensuring output meets minimum standards |

## 2. Core Entities

### 2.1 Idea Entity

```python
class Idea:
    # Identity
    idea_id: UUID (Primary Key)
    created_at: DateTime
    updated_at: DateTime
    version: Integer
    
    # Core Attributes
    title: String (required, 10-200 chars)
    description: String (required, 10-5000 chars)
    category: IdeaCategory (enum)
    status: IdeaStatus (enum)
    
    # Content
    problem_statement: String (optional, max 1000 chars)
    solution_description: String (optional, max 2000 chars)
    target_market: String (optional, max 1000 chars)
    competitive_advantage: String (optional, max 1000 chars)
    
    # Workflow State
    current_stage: PipelineStage (enum)
    stage_progress: Float (0.0-1.0)
    processing_started_at: DateTime (nullable)
    processing_completed_at: DateTime (nullable)
    
    # Quality Metrics
    evidence_score: Float (0.0-1.0, nullable)
    investor_score: Float (0.0-1.0, nullable)
    smoke_test_score: Float (0.0-1.0, nullable)
    
    # File References
    deck_path: String (nullable, file system path)
    mvp_repository_url: String (nullable, git URL)
    landing_page_url: String (nullable, deployed URL)
    
    # Relationships
    evidence_items: List[Evidence]
    investor_evaluations: List[InvestorEvaluation]
    smoke_test_campaigns: List[SmokeTestCampaign]
    audit_trail: List[IdeaAuditEntry]
```

**Business Rules:**
- Ideas must have unique titles within the same category
- Status transitions must follow defined workflow stages
- Evidence score requires minimum 3 citations
- Ideas cannot be deleted, only archived

**Invariants:**
- `created_at <= updated_at`
- `stage_progress` must be between 0.0 and 1.0
- `current_stage` must match the furthest completed workflow stage

### 2.2 Evidence Entity

```python
class Evidence:
    # Identity
    evidence_id: UUID (Primary Key)
    idea_id: UUID (Foreign Key -> Idea)
    created_at: DateTime
    
    # Content
    claim_text: String (required, max 500 chars)
    citation_url: String (required, valid URL)
    citation_title: String (required, max 200 chars)
    citation_source: String (required, max 100 chars)
    relevance_score: Float (0.0-1.0)
    
    # Classification
    evidence_type: EvidenceType (enum)
    research_domain: ResearchDomain (enum)
    verification_status: VerificationStatus (enum)
    
    # Quality Metrics
    credibility_score: Float (0.0-1.0)
    recency_score: Float (0.0-1.0)
    accessibility_verified: Boolean
    
    # Agent Attribution
    collected_by_agent: String (agent identifier)
    collection_method: String (RAG, API, manual)
```

**Business Rules:**
- Each evidence item must support exactly one claim
- Citations must be publicly accessible
- Evidence items older than 2 years have reduced recency scores
- Minimum 3 evidence items required per idea for progression

### 2.3 InvestorEvaluation Entity

```python
class InvestorEvaluation:
    # Identity
    evaluation_id: UUID (Primary Key)
    idea_id: UUID (Foreign Key -> Idea)
    evaluator_agent: String (VC, Angel, etc.)
    created_at: DateTime
    
    # Scoring Components (weighted)
    team_score: Float (0.0-1.0)
    market_score: Float (0.0-1.0)
    tech_moat_score: Float (0.0-1.0)
    evidence_score: Float (0.0-1.0)
    
    # Composite Scores
    weighted_total_score: Float (0.0-1.0)
    funding_recommendation: FundingDecision (enum)
    confidence_level: Float (0.0-1.0)
    
    # Feedback
    strengths: List[String]
    concerns: List[String]
    improvement_suggestions: List[String]
    
    # Rubric Reference
    rubric_version: String
    rubric_weights: Dict[String, Float]
```

**Business Rules:**
- All score components must be present for valid evaluation
- Weighted total must match calculated score from components
- Funding recommendation threshold configurable via environment
- Each idea requires evaluation from minimum 2 different investor types

### 2.4 SmokeTestCampaign Entity

```python
class SmokeTestCampaign:
    # Identity
    campaign_id: UUID (Primary Key)
    idea_id: UUID (Foreign Key -> Idea)
    created_at: DateTime
    launched_at: DateTime (nullable)
    ended_at: DateTime (nullable)
    
    # Campaign Configuration
    campaign_name: String (required, max 100 chars)
    budget_allocated: Decimal (currency amount)
    budget_spent: Decimal (currency amount, default 0)
    target_audience: String (demographic description)
    
    # Landing Page
    landing_page_url: String (deployed URL)
    cta_type: CTAType (enum: signup, purchase, contact)
    lighthouse_score: Integer (0-100, nullable)
    
    # Performance Metrics
    impressions: Integer (default 0)
    clicks: Integer (default 0)
    conversions: Integer (default 0)
    click_through_rate: Float (calculated)
    conversion_rate: Float (calculated)
    cost_per_click: Decimal (calculated)
    cost_per_conversion: Decimal (calculated)
    
    # Status
    campaign_status: CampaignStatus (enum)
    quality_gates_passed: Boolean (default False)
    
    # Analytics Integration
    posthog_project_id: String (nullable)
    analytics_events: List[AnalyticsEvent]
```

**Business Rules:**
- Budget spent cannot exceed budget allocated
- Campaign cannot launch without valid landing page (Lighthouse >90)
- Automatic pause when budget reaches configured threshold ($50)
- Minimum 100 impressions required for meaningful metrics

### 2.5 MVPArtifact Entity

```python
class MVPArtifact:
    # Identity
    mvp_id: UUID (Primary Key)
    idea_id: UUID (Foreign Key -> Idea)
    created_at: DateTime
    
    # Repository Information
    repository_url: String (git URL)
    repository_branch: String (default "main")
    commit_hash: String (git commit SHA)
    
    # Generation Details
    generator_tool: String (GPT-Engineer, etc.)
    generation_prompt: String (max 2000 chars)
    technology_stack: List[String]
    
    # Quality Metrics
    test_coverage_percentage: Float (0.0-100.0)
    linting_passed: Boolean
    security_scan_passed: Boolean
    build_status: BuildStatus (enum)
    
    # Deployment
    deployment_url: String (nullable, live URL)
    deployment_platform: String (Fly.io, etc.)
    health_check_status: HealthStatus (enum)
    uptime_percentage: Float (0.0-100.0)
    
    # Performance
    response_time_ms: Integer (nullable)
    error_rate: Float (0.0-1.0)
    last_health_check: DateTime
```

**Business Rules:**
- Repository must be accessible for deployment
- Health checks required every 15 minutes
- Deployment fails if test coverage <90%
- Automatic rollback if error rate >5%

## 3. Value Objects

### 3.1 BudgetThreshold
```python
class BudgetThreshold:
    token_budget_usd: Decimal (max spend per idea)
    ad_budget_usd: Decimal (max ad spend per campaign)
    alert_threshold_percentage: Float (when to send warnings)
    auto_pause_enabled: Boolean
```

### 3.2 QualityGate
```python
class QualityGate:
    gate_name: String
    minimum_score: Float (0.0-1.0)
    required_criteria: List[String]
    bypass_allowed: Boolean (for manual override)
```

### 3.3 AgentConfiguration
```python
class AgentConfiguration:
    agent_type: String (CEO, CTO, VC, etc.)
    model_name: String (GPT-4, Gemini-2.5-Pro)
    temperature: Float (0.0-2.0)
    max_tokens: Integer
    system_prompt: String
    tools_available: List[String]
```

## 4. Enumerations

### 4.1 Pipeline States
```python
class PipelineStage(Enum):
    IDEATE = "ideate"
    RESEARCH = "research"  
    DECK_GENERATION = "deck_generation"
    INVESTOR_EVALUATION = "investor_evaluation"
    SMOKE_TEST = "smoke_test"
    MVP_GENERATION = "mvp_generation"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
```

### 4.2 Status Enumerations
```python
class IdeaStatus(Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    READY_FOR_EVALUATION = "ready_for_evaluation"
    FUNDED = "funded"
    REJECTED = "rejected"
    SMOKE_TEST_PASSED = "smoke_test_passed"
    SMOKE_TEST_FAILED = "smoke_test_failed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"

class VerificationStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    INACCESSIBLE = "inaccessible"

class FundingDecision(Enum):
    STRONG_YES = "strong_yes"
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"
    STRONG_NO = "strong_no"
```

## 5. Aggregates and Boundaries

### 5.1 Idea Aggregate
**Root Entity:** Idea
**Contained Entities:** Evidence, InvestorEvaluation, SmokeTestCampaign, MVPArtifact
**Boundary:** Complete idea lifecycle from creation to deployment
**Consistency Rules:**
- Idea status must reflect furthest completed stage
- Evidence score aggregated from all evidence items
- Cannot progress to next stage without meeting quality gates

### 5.2 Campaign Aggregate  
**Root Entity:** SmokeTestCampaign
**Contained Entities:** AnalyticsEvent, BudgetSpendEvent
**Boundary:** Single marketing campaign lifecycle
**Consistency Rules:**
- Budget spent cannot exceed allocation
- Campaign metrics must be recalculated on each update
- Analytics events must be chronologically ordered

## 6. Domain Events

### 6.1 Idea Lifecycle Events
```python
class IdeaCreated(DomainEvent):
    idea_id: UUID
    title: String
    created_by: String
    timestamp: DateTime

class IdeaStageProgressed(DomainEvent):
    idea_id: UUID
    from_stage: PipelineStage
    to_stage: PipelineStage
    triggered_by: String
    timestamp: DateTime

class QualityGateFailed(DomainEvent):
    idea_id: UUID
    gate_name: String
    current_score: Float
    required_score: Float
    timestamp: DateTime
```

### 6.2 Budget Control Events
```python
class BudgetThresholdExceeded(DomainEvent):
    entity_id: UUID
    entity_type: String (idea, campaign)
    budget_type: String (token, ad)
    spent_amount: Decimal
    threshold_amount: Decimal
    auto_paused: Boolean
    timestamp: DateTime
```

## 7. Repository Interfaces

### 7.1 Core Repositories
```python
class IdeaRepository:
    def save(idea: Idea) -> None
    def find_by_id(idea_id: UUID) -> Optional[Idea]
    def find_by_status(status: IdeaStatus) -> List[Idea]
    def find_similar_ideas(description: String, threshold: Float) -> List[Idea]
    def update_stage_progress(idea_id: UUID, stage: PipelineStage, progress: Float) -> None

class EvidenceRepository:
    def save(evidence: Evidence) -> None
    def find_by_idea_id(idea_id: UUID) -> List[Evidence]
    def verify_citation_accessibility(evidence_id: UUID) -> Boolean
    def calculate_evidence_score(idea_id: UUID) -> Float

class CampaignRepository:
    def save(campaign: SmokeTestCampaign) -> None
    def find_active_campaigns() -> List[SmokeTestCampaign]
    def update_metrics(campaign_id: UUID, metrics: Dict) -> None
    def pause_campaign(campaign_id: UUID, reason: String) -> None
```

## 8. Domain Services

### 8.1 Workflow Orchestration Service
```python
class PipelineOrchestrator:
    def advance_to_next_stage(idea_id: UUID) -> Result[PipelineStage, Error]
    def validate_stage_requirements(idea_id: UUID, target_stage: PipelineStage) -> ValidationResult
    def calculate_overall_progress(idea_id: UUID) -> Float
    def handle_stage_failure(idea_id: UUID, error: Error) -> None
```

### 8.2 Quality Assessment Service
```python
class QualityAssessmentService:
    def evaluate_evidence_quality(evidence_items: List[Evidence]) -> Float
    def assess_investor_consensus(evaluations: List[InvestorEvaluation]) -> ConsensusResult
    def validate_smoke_test_readiness(idea_id: UUID) -> ValidationResult
    def check_quality_gates(idea_id: UUID, stage: PipelineStage) -> List[QualityGateResult]
```

### 8.3 Budget Control Service
```python
class BudgetControlService:
    def check_budget_limits(entity_id: UUID, requested_amount: Decimal) -> BudgetCheckResult
    def record_spending(entity_id: UUID, amount: Decimal, category: String) -> None
    def pause_entity_on_budget_exceeded(entity_id: UUID) -> None
    def generate_budget_alerts() -> List[BudgetAlert]
```

## 9. Integration Contracts

### 9.1 External Service Interfaces
```python
class ExternalAPIContract:
    # Google Ads Integration
    def create_ad_campaign(campaign_config: Dict) -> Result[String, Error]
    def get_campaign_metrics(campaign_id: String) -> Result[Dict, Error]
    def pause_campaign(campaign_id: String) -> Result[None, Error]
    
    # PostHog Analytics
    def track_event(event_name: String, properties: Dict) -> Result[None, Error]
    def get_conversion_metrics(project_id: String, date_range: Tuple) -> Result[Dict, Error]
    
    # Deployment Platform (Fly.io)
    def deploy_application(app_config: Dict) -> Result[String, Error]
    def get_health_status(app_id: String) -> Result[HealthStatus, Error]
    def scale_application(app_id: String, instances: Integer) -> Result[None, Error]
```

## 10. Data Consistency Rules

### 10.1 Cross-Aggregate Consistency
- **Idea ↔ Evidence**: Evidence score must be recalculated when evidence items are added/modified
- **Idea ↔ Campaign**: Smoke test score must be updated when campaign metrics change  
- **Idea ↔ MVP**: Deployment status must reflect actual MVP health check results

### 10.2 Eventually Consistent Updates
- Analytics metrics may be updated asynchronously (5-minute delay acceptable)
- Budget spending records may have temporary inconsistencies during high-throughput periods
- Health check statuses updated every 15 minutes, not real-time

### 10.3 Strong Consistency Requirements  
- Budget limits must be enforced immediately
- Quality gate validation must be synchronous
- Stage progression must be atomic (all-or-nothing)