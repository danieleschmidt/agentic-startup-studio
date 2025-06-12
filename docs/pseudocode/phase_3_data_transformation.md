# Phase 3: Data Transformation - Pseudocode Specification

## Module Overview
**Responsibility:** Transform research data into pitch decks, orchestrate workflow state machine, and conduct investor evaluations
**Input:** Research results with evidence and quality scores
**Output:** Generated pitch decks, investor evaluations, and state transitions
**Quality Gates:** Deck accessibility standards, investor scoring thresholds, state validation

## 1. LangGraph State Machine Orchestrator

### 1.1 Pipeline State Management
```pseudocode
FUNCTION execute_pipeline_state_machine(idea_id: UUID) -> Result[PipelineExecution, Error]:
    // TEST: should enforce proper state transitions (Ideate→Research→Deck→Investors)
    // TEST: should handle state rollbacks on quality gate failures
    // TEST: should support checkpoint/resume functionality for long-running workflows
    // TEST: should emit state change events for monitoring and debugging
    
    execution_context = PipelineExecutionContext(
        idea_id: idea_id,
        started_at: get_current_timestamp(),
        current_state: get_idea_current_stage(idea_id),
        checkpoint_enabled: get_config("ENABLE_CHECKPOINTS") OR true
    )
    
    state_machine = create_langgraph_state_machine(execution_context)
    
    try:
        // Execute state machine with monitoring
        execution_result = state_machine.execute_with_monitoring(
            timeout: get_config("PIPELINE_TIMEOUT") OR 1800,  // 30 minutes
            checkpoint_interval: get_config("CHECKPOINT_INTERVAL") OR 300  // 5 minutes
        )
        
        // Final validation and completion
        final_state = execution_result.final_state
        IF final_state.is_terminal_success():
            mark_pipeline_completed(idea_id, execution_result)
            publish_pipeline_completion_event(idea_id, execution_result)
        ELSE IF final_state.is_failure():
            handle_pipeline_failure(idea_id, execution_result)
        
        RETURN Success(execution_result)
        
    catch PipelineTimeoutError as e:
        // Save checkpoint for resume
        save_execution_checkpoint(execution_context, state_machine.get_current_state())
        logger.error("Pipeline execution timed out", idea_id: idea_id, error: e)
        RETURN Error("Pipeline execution timed out: " + e.message)
    
    catch StateTransitionError as e:
        // Attempt recovery or rollback
        recovery_result = attempt_state_recovery(execution_context, e)
        IF recovery_result.is_success():
            RETURN execute_pipeline_state_machine(idea_id)  // Retry after recovery
        ELSE:
            RETURN Error("State transition failed: " + e.message)

FUNCTION create_langgraph_state_machine(context: PipelineExecutionContext) -> LangGraphStateMachine:
    // TEST: should define all required states and transitions
    // TEST: should include quality gates at each transition
    // TEST: should support conditional branching based on scores
    // TEST: should handle agent task failures with retries
    
    state_machine = LangGraphStateMachine()
    
    // Define states
    state_machine.add_state("ideate", IdeateState())
    state_machine.add_state("research", ResearchState()) 
    state_machine.add_state("deck_generation", DeckGenerationState())
    state_machine.add_state("investor_evaluation", InvestorEvaluationState())
    state_machine.add_state("smoke_test_prep", SmokeTestPrepState())
    state_machine.add_state("completed", CompletedState())
    state_machine.add_state("failed", FailedState())
    
    // Define transitions with quality gates
    state_machine.add_transition(
        from_state: "ideate",
        to_state: "research", 
        condition: validate_idea_completeness,
        quality_gate: check_idea_quality_gate
    )
    
    state_machine.add_transition(
        from_state: "research",
        to_state: "deck_generation",
        condition: validate_evidence_requirements,
        quality_gate: check_evidence_quality_gate
    )
    
    state_machine.add_transition(
        from_state: "deck_generation", 
        to_state: "investor_evaluation",
        condition: validate_deck_accessibility,
        quality_gate: check_deck_quality_gate
    )
    
    state_machine.add_transition(
        from_state: "investor_evaluation",
        to_state: "smoke_test_prep",
        condition: validate_funding_threshold,
        quality_gate: check_investor_consensus_gate
    )
    
    state_machine.add_transition(
        from_state: "investor_evaluation",
        to_state: "failed", 
        condition: below_funding_threshold,
        quality_gate: None  // No gate needed for failure transition
    )
    
    // Configure error handling and retry policies
    state_machine.set_retry_policy(
        max_retries: get_config("MAX_STATE_RETRIES") OR 3,
        backoff_strategy: "exponential",
        retryable_errors: [TemporaryError, NetworkError, ResourceUnavailableError]
    )
    
    RETURN state_machine
```

## 2. Pitch Deck Generation Module

### 2.1 Marp Deck Generator
```pseudocode
FUNCTION generate_pitch_deck(idea_id: UUID, research_results: ResearchResults) -> Result[DeckGeneration, Error]:
    // TEST: should generate 10-slide deck following standardized template
    // TEST: should populate content from structured idea and research data
    // TEST: should achieve Lighthouse accessibility score >90
    // TEST: should handle missing or incomplete research gracefully
    
    idea = idea_repository.find_by_id(idea_id)
    deck_template = load_deck_template()
    
    // Gather content for deck generation
    deck_content = compile_deck_content(idea, research_results)
    validation_result = validate_deck_content_completeness(deck_content)
    
    IF NOT validation_result.is_complete():
        // Attempt to fill gaps with default content or research
        deck_content = fill_content_gaps(deck_content, validation_result.missing_sections)
    
    // Generate Marp-compatible markdown
    marp_content = generate_marp_markdown(deck_content, deck_template)
    // TEST: should produce valid Marp syntax
    
    // Validate generated content
    syntax_validation = validate_marp_syntax(marp_content)
    IF NOT syntax_validation.is_valid():
        RETURN Error("Invalid Marp syntax: " + syntax_validation.errors.join(", "))
    
    // Render deck to HTML for accessibility testing
    html_deck = render_marp_to_html(marp_content)
    accessibility_score = test_accessibility(html_deck)
    
    IF accessibility_score < get_config("MIN_DECK_ACCESSIBILITY") OR 90:
        // Attempt to fix accessibility issues
        fixed_content = fix_accessibility_issues(marp_content, html_deck)
        html_deck = render_marp_to_html(fixed_content)
        accessibility_score = test_accessibility(html_deck)
        
        IF accessibility_score < get_config("MIN_DECK_ACCESSIBILITY") OR 90:
            logger.warn("Deck accessibility below threshold", 
                       idea_id: idea_id, score: accessibility_score)
    
    // Save deck files
    deck_path = generate_deck_file_path(idea_id)
    save_deck_files(deck_path, marp_content, html_deck)
    
    // Update idea with deck information
    update_idea_with_deck_info(idea_id, deck_path, accessibility_score)
    
    deck_generation = DeckGeneration(
        idea_id: idea_id,
        deck_path: deck_path,
        accessibility_score: accessibility_score,
        slide_count: count_slides(marp_content),
        content_completeness: validation_result.completeness_score,
        generated_at: get_current_timestamp()
    )
    
    RETURN Success(deck_generation)

FUNCTION compile_deck_content(idea: Idea, research: ResearchResults) -> DeckContent:
    // TEST: should extract key information for each slide type
    // TEST: should prioritize high-credibility evidence for claims
    // TEST: should handle missing sections with appropriate defaults
    // TEST: should maintain consistent narrative flow across slides
    
    evidence_by_domain = group_evidence_by_domain(research.evidence_items)
    top_evidence = select_top_evidence_items(evidence_by_domain, max_per_domain: 3)
    
    deck_content = DeckContent(
        // Slide 1: Title & Tagline
        title: idea.title,
        tagline: generate_compelling_tagline(idea),
        presenter_info: get_presenter_info(),
        
        // Slide 2: Problem Statement  
        problem_statement: idea.problem_statement OR generate_problem_from_description(idea),
        problem_evidence: filter_evidence_by_type(top_evidence, EvidenceType.PROBLEM_VALIDATION),
        
        // Slide 3: Solution Overview
        solution_description: idea.solution_description OR extract_solution_from_description(idea),
        unique_value_proposition: generate_value_proposition(idea, research),
        
        // Slide 4: Market Opportunity
        target_market: idea.target_market,
        market_size_data: extract_market_data(evidence_by_domain[ResearchDomain.MARKET_ANALYSIS]),
        market_trends: extract_market_trends(evidence_by_domain[ResearchDomain.MARKET_ANALYSIS]),
        
        // Slide 5: Competitive Landscape
        competitive_analysis: compile_competitive_analysis(evidence_by_domain[ResearchDomain.COMPETITIVE_LANDSCAPE]),
        competitive_advantages: identify_competitive_advantages(idea, research),
        
        // Slide 6: Technical Approach (if applicable)
        technical_feasibility: extract_technical_insights(evidence_by_domain[ResearchDomain.TECHNICAL_FEASIBILITY]),
        technology_stack: suggest_technology_stack(idea),
        
        // Slide 7: Business Model
        revenue_model: suggest_revenue_model(idea, research),
        pricing_strategy: suggest_pricing_strategy(research),
        
        // Slide 8: Go-to-Market Strategy
        marketing_channels: suggest_marketing_channels(idea.target_market),
        customer_acquisition: suggest_acquisition_strategy(research),
        
        // Slide 9: Financial Projections
        financial_assumptions: generate_financial_assumptions(idea, research),
        revenue_projections: generate_revenue_projections(idea),
        
        // Slide 10: Funding & Next Steps
        funding_request: calculate_funding_request(idea),
        milestones: generate_key_milestones(idea),
        contact_information: get_contact_information()
    )
    
    RETURN deck_content
```

## 3. Investor Evaluation System

### 3.1 Multi-Agent Investor Scoring
```pseudocode
FUNCTION execute_investor_evaluation(idea_id: UUID, deck_path: String) -> Result[InvestorEvaluationResults, Error]:
    // TEST: should evaluate idea using multiple investor agent types (VC, Angel)
    // TEST: should apply weighted scoring rubric consistently
    // TEST: should require minimum number of evaluations for consensus
    // TEST: should handle evaluator disagreements with conflict resolution
    
    idea = idea_repository.find_by_id(idea_id)
    research_results = get_research_results(idea_id)
    deck_content = load_deck_content(deck_path)
    
    // Configure investor agents
    investor_agents = configure_investor_agents()
    evaluation_context = InvestorEvaluationContext(
        idea: idea,
        research_results: research_results,
        deck_content: deck_content,
        rubric: load_evaluation_rubric()
    )
    
    // Execute parallel evaluations
    evaluation_tasks = []
    FOR agent IN investor_agents:
        task = create_evaluation_task(agent, evaluation_context)
        evaluation_tasks.append(task)
    
    // TEST: should complete evaluations within timeout period
    task_results = execute_parallel_evaluations(
        tasks: evaluation_tasks,
        timeout: get_config("EVALUATION_TIMEOUT") OR 600  // 10 minutes
    )
    
    // Process and validate evaluation results
    successful_evaluations = []
    failed_evaluations = []
    
    FOR agent, result IN task_results.items():
        IF result.is_success():
            validated_evaluation = validate_evaluation_result(result.evaluation)
            IF validated_evaluation.is_valid():
                successful_evaluations.append(validated_evaluation)
            ELSE:
                logger.warn("Invalid evaluation from agent", agent: agent.type, 
                          validation_errors: validated_evaluation.errors)
                failed_evaluations.append(agent)
        ELSE:
            logger.error("Evaluation failed for agent", agent: agent.type, error: result.error)
            failed_evaluations.append(agent)
    
    // Check minimum evaluation requirements
    min_evaluations = get_config("MIN_INVESTOR_EVALUATIONS") OR 2
    IF successful_evaluations.length < min_evaluations:
        // Attempt to get additional evaluations
        retry_result = retry_failed_evaluations(failed_evaluations, evaluation_context)
        successful_evaluations.extend(retry_result.additional_evaluations)
    
    IF successful_evaluations.length < min_evaluations:
        RETURN Error("Insufficient investor evaluations: " + successful_evaluations.length + " < " + min_evaluations)
    
    // Calculate consensus and final scores
    consensus_analysis = analyze_evaluation_consensus(successful_evaluations)
    final_scores = calculate_final_scores(successful_evaluations, consensus_analysis)
    funding_recommendation = determine_funding_recommendation(final_scores)
    
    evaluation_results = InvestorEvaluationResults(
        idea_id: idea_id,
        evaluations: successful_evaluations,
        consensus_analysis: consensus_analysis,
        final_scores: final_scores,
        funding_recommendation: funding_recommendation,
        completed_at: get_current_timestamp()
    )
    
    // Save results and update idea
    save_evaluation_results(evaluation_results)
    update_idea_with_evaluation(idea_id, evaluation_results)
    
    RETURN Success(evaluation_results)

FUNCTION create_evaluation_task(agent: InvestorAgent, context: InvestorEvaluationContext) -> EvaluationTask:
    // TEST: should create agent-specific evaluation prompts
    // TEST: should include rubric weights in agent configuration
    // TEST: should provide complete context for informed evaluation
    // TEST: should set appropriate model parameters for consistent scoring
    
    agent_prompt = build_agent_evaluation_prompt(agent, context)
    
    evaluation_task = EvaluationTask(
        agent: agent,
        prompt: agent_prompt,
        model_config: ModelConfig(
            model_name: agent.preferred_model,
            temperature: get_config("EVALUATION_TEMPERATURE") OR 0.3,  // Low for consistency
            max_tokens: get_config("EVALUATION_MAX_TOKENS") OR 2000,
            top_p: 0.9
        ),
        rubric: context.rubric,
        timeout: get_config("SINGLE_EVALUATION_TIMEOUT") OR 120  // 2 minutes
    )
    
    RETURN evaluation_task
```

## 4. Scoring Rubric Engine

### 4.1 Weighted Evaluation System
```pseudocode
FUNCTION calculate_weighted_scores(evaluation: InvestorEvaluation, rubric: EvaluationRubric) -> WeightedScores:
    // TEST: should apply correct weights for each scoring dimension
    // TEST: should normalize scores to 0-1 range
    // TEST: should handle missing scores gracefully
    // TEST: should validate rubric weights sum to 1.0
    
    // Validate rubric configuration
    rubric_validation = validate_rubric_weights(rubric)
    IF NOT rubric_validation.is_valid():
        THROW InvalidRubricError("Rubric weights invalid: " + rubric_validation.errors.join(", "))
    
    // Extract dimension scores
    dimension_scores = DimensionScores(
        team_score: evaluation.team_score OR 0.0,
        market_score: evaluation.market_score OR 0.0, 
        tech_moat_score: evaluation.tech_moat_score OR 0.0,
        evidence_score: evaluation.evidence_score OR 0.0
    )
    
    // Apply weights and calculate composite score
    weighted_total = (
        dimension_scores.team_score * rubric.team_weight +
        dimension_scores.market_score * rubric.market_weight +
        dimension_scores.tech_moat_score * rubric.tech_moat_weight + 
        dimension_scores.evidence_score * rubric.evidence_weight
    )
    
    // Confidence adjustment based on score variance
    score_variance = calculate_dimension_score_variance(dimension_scores)
    confidence_factor = calculate_confidence_factor(score_variance)
    adjusted_score = weighted_total * confidence_factor
    
    weighted_scores = WeightedScores(
        composite_score: adjusted_score,
        dimension_scores: dimension_scores,
        applied_weights: extract_applied_weights(rubric),
        confidence_factor: confidence_factor,
        score_variance: score_variance
    )
    
    RETURN weighted_scores

FUNCTION determine_funding_recommendation(final_scores: FinalScores) -> FundingRecommendation:
    // TEST: should use configurable funding threshold
    // TEST: should consider consensus level in recommendation
    // TEST: should provide reasoning for recommendation
    // TEST: should handle edge cases near threshold boundaries
    
    funding_threshold = get_config("FUNDING_THRESHOLD") OR 0.8
    consensus_threshold = get_config("CONSENSUS_THRESHOLD") OR 0.7
    
    // Base recommendation from composite score
    base_recommendation = final_scores.composite_score >= funding_threshold ? 
                         FundingDecision.YES : FundingDecision.NO
    
    // Adjust based on consensus level
    consensus_adjustment = calculate_consensus_adjustment(
        consensus_level: final_scores.consensus_level,
        consensus_threshold: consensus_threshold
    )
    
    // Apply confidence penalties
    confidence_penalty = calculate_confidence_penalty(final_scores.confidence_metrics)
    
    final_recommendation = apply_recommendation_adjustments(
        base_recommendation: base_recommendation,
        consensus_adjustment: consensus_adjustment,
        confidence_penalty: confidence_penalty
    )
    
    // Generate reasoning
    recommendation_reasoning = generate_recommendation_reasoning(
        final_scores: final_scores,
        threshold: funding_threshold,
        adjustments: [consensus_adjustment, confidence_penalty]
    )
    
    funding_recommendation = FundingRecommendation(
        decision: final_recommendation,
        confidence_level: final_scores.confidence_metrics.overall_confidence,
        composite_score: final_scores.composite_score,
        consensus_level: final_scores.consensus_level,
        reasoning: recommendation_reasoning,
        threshold_used: funding_threshold,
        recommendation_date: get_current_timestamp()
    )
    
    RETURN funding_recommendation
```

## 5. Quality Gate Validation

### 5.1 Stage Transition Gates
```pseudocode
FUNCTION validate_research_to_deck_gate(idea_id: UUID) -> QualityGateResult:
    // TEST: should check evidence quality requirements
    // TEST: should verify minimum citation count
    // TEST: should validate evidence accessibility
    // TEST: should allow bypass with manual approval
    
    research_results = get_research_results(idea_id)
    evidence_score = research_results.overall_score
    evidence_count = research_results.evidence_items.length
    
    gate_requirements = load_quality_gate_requirements("research_to_deck")
    validation_results = []
    
    // Check evidence score requirement
    IF evidence_score >= gate_requirements.min_evidence_score:
        validation_results.append(ValidationResult("evidence_score", true, evidence_score))
    ELSE:
        validation_results.append(ValidationResult("evidence_score", false, 
                                                   "Score " + evidence_score + " below threshold " + gate_requirements.min_evidence_score))
    
    // Check evidence count requirement  
    IF evidence_count >= gate_requirements.min_evidence_count:
        validation_results.append(ValidationResult("evidence_count", true, evidence_count))
    ELSE:
        validation_results.append(ValidationResult("evidence_count", false,
                                                   "Count " + evidence_count + " below minimum " + gate_requirements.min_evidence_count))
    
    // Check accessibility rate
    accessibility_rate = calculate_evidence_accessibility_rate(research_results.evidence_items)
    IF accessibility_rate >= gate_requirements.min_accessibility_rate:
        validation_results.append(ValidationResult("accessibility_rate", true, accessibility_rate))
    ELSE:
        validation_results.append(ValidationResult("accessibility_rate", false,
                                                   "Rate " + accessibility_rate + " below threshold " + gate_requirements.min_accessibility_rate))
    
    // Calculate overall gate result
    passed_count = count_passed_validations(validation_results)
    total_count = validation_results.length
    gate_passed = passed_count == total_count
    
    quality_gate_result = QualityGateResult(
        gate_name: "research_to_deck",
        idea_id: idea_id,
        passed: gate_passed,
        validation_results: validation_results,
        overall_score: passed_count / total_count,
        bypass_available: gate_requirements.allow_bypass,
        checked_at: get_current_timestamp()
    )
    
    // Log gate validation
    log_quality_gate_check(quality_gate_result)
    
    RETURN quality_gate_result

FUNCTION validate_deck_to_investor_gate(idea_id: UUID) -> QualityGateResult:
    // TEST: should verify deck accessibility standards
    // TEST: should check deck completeness
    // TEST: should validate slide count and content structure
    // TEST: should ensure deck file accessibility
    
    idea = idea_repository.find_by_id(idea_id)
    deck_path = idea.deck_path
    
    IF deck_path IS NULL OR NOT file_exists(deck_path):
        RETURN QualityGateResult(
            gate_name: "deck_to_investor",
            idea_id: idea_id,
            passed: false,
            validation_results: [ValidationResult("deck_exists", false, "Deck file not found")],
            overall_score: 0.0
        )
    
    gate_requirements = load_quality_gate_requirements("deck_to_investor")
    validation_results = []
    
    // Check deck accessibility
    deck_html = load_deck_html(deck_path)
    accessibility_score = test_accessibility(deck_html)
    
    IF accessibility_score >= gate_requirements.min_accessibility_score:
        validation_results.append(ValidationResult("accessibility", true, accessibility_score))
    ELSE:
        validation_results.append(ValidationResult("accessibility", false,
                                                   "Accessibility score " + accessibility_score + " below " + gate_requirements.min_accessibility_score))
    
    // Check slide count
    slide_count = count_slides_in_deck(deck_path)
    IF slide_count >= gate_requirements.min_slide_count AND slide_count <= gate_requirements.max_slide_count:
        validation_results.append(ValidationResult("slide_count", true, slide_count))
    ELSE:
        validation_results.append(ValidationResult("slide_count", false,
                                                   "Slide count " + slide_count + " outside range [" + 
                                                   gate_requirements.min_slide_count + ", " + gate_requirements.max_slide_count + "]"))
    
    // Check content completeness
    content_completeness = analyze_deck_content_completeness(deck_path)
    IF content_completeness >= gate_requirements.min_content_completeness:
        validation_results.append(ValidationResult("content_completeness", true, content_completeness))
    ELSE:
        validation_results.append(ValidationResult("content_completeness", false,
                                                   "Content completeness " + content_completeness + " below " + gate_requirements.min_content_completeness))
    
    // Calculate overall result
    passed_count = count_passed_validations(validation_results)
    total_count = validation_results.length
    gate_passed = passed_count == total_count
    
    RETURN QualityGateResult(
        gate_name: "deck_to_investor",
        idea_id: idea_id,
        passed: gate_passed,
        validation_results: validation_results,
        overall_score: passed_count / total_count,
        checked_at: get_current_timestamp()
    )
```

## 6. State Persistence and Recovery

### 6.1 Checkpoint Management
```pseudocode
FUNCTION save_execution_checkpoint(context: PipelineExecutionContext, current_state: String) -> None:
    // TEST: should save complete execution state for recovery
    // TEST: should include timestamps and progress metrics  
    // TEST: should handle checkpoint storage failures gracefully
    // TEST: should compress checkpoint data for efficient storage
    
    checkpoint_data = ExecutionCheckpoint(
        idea_id: context.idea_id,
        execution_id: context.execution_id,
        current_state: current_state,
        started_at: context.started_at,
        checkpointed_at: get_current_timestamp(),
        progress_metrics: context.progress_metrics,
        completed_states: context.completed_states,
        state_context: serialize_state_context(context.state_context),
        error_history: context.error_history
    )
    
    try:
        compressed_data = compress_checkpoint_data(checkpoint_data)
        checkpoint_repository.save(context.idea_id, compressed_data)
        logger.info("Checkpoint saved", idea_id: context.idea_id, state: current_state)
    catch CheckpointStorageError as e:
        logger.error("Failed to save checkpoint", idea_id: context.idea_id, error: e)
        // Don't fail the main execution, but record the issue
        metrics.counter("checkpoint_failures").increment()

FUNCTION resume_from_checkpoint(idea_id: UUID) -> Result[PipelineExecutionContext, Error]:
    // TEST: should restore execution state from last checkpoint
    // TEST: should validate checkpoint integrity before resuming
    // TEST: should handle corrupted or missing checkpoints
    // TEST: should update context with current timestamp
    
    checkpoint_data = checkpoint_repository.find_latest(idea_id)
    IF checkpoint_data IS NULL:
        RETURN Error("No checkpoint found for idea: " + idea_id)
    
    try:
        decompressed_data = decompress_checkpoint_data(checkpoint_data)
        checkpoint = deserialize_checkpoint(decompressed_data)
        
        // Validate checkpoint integrity
        validation_result = validate_checkpoint_integrity(checkpoint)
        IF NOT validation_result.is_valid():
            RETURN Error("Checkpoint validation failed: " + validation_result.errors.join(", "))
        
        // Restore execution context
        restored_context = restore_execution_context(checkpoint)
        restored_context.resumed_at = get_current_timestamp()
        restored_context.resume_count += 1
        
        logger.info("Execution resumed from checkpoint", 
                   idea_id: idea_id, 
                   checkpoint_state: checkpoint.current_state,
                   resume_count: restored_context.resume_count)
        
        RETURN Success(restored_context)
        
    catch CheckpointCorruptionError as e:
        logger.error("Checkpoint data corrupted", idea_id: idea_id, error: e)
        RETURN Error("Checkpoint data corrupted: " + e.message)
```

---

**Summary**: This module handles the core transformation workflow through LangGraph state machine orchestration, pitch deck generation using Marp templates, multi-agent investor evaluation with weighted scoring, and robust quality gates. All components include comprehensive TDD anchors and error handling for production reliability.