# Phase 2: Data Processing - Pseudocode Specification

## Module Overview
**Responsibility:** Collect, verify, and score evidence for startup ideas through automated research agents
**Input:** Validated Idea entities requiring research
**Output:** Evidence collection with quality scores and citations
**Quality Gates:** Minimum 3 citations per claim, accessibility verification, credibility scoring

## 1. Evidence Collection Orchestrator

### 1.1 Research Workflow Controller
```pseudocode
FUNCTION execute_research_workflow(idea_id: UUID) -> Result[ResearchResults, Error]:
    // TEST: should coordinate all research agents for comprehensive analysis
    // TEST: should handle partial failures gracefully and continue with available data
    // TEST: should enforce minimum evidence requirements before completion
    // TEST: should update idea stage progress throughout research process
    
    idea = idea_repository.find_by_id(idea_id)
    IF idea IS NULL:
        RETURN Error("Idea not found: " + idea_id)
    
    IF idea.current_stage != PipelineStage.RESEARCH:
        RETURN Error("Idea not in research stage: " + idea.current_stage)
    
    // Initialize research context
    research_context = ResearchContext(
        idea_id: idea_id,
        title: idea.title,
        description: idea.description,
        target_market: idea.target_market,
        problem_statement: idea.problem_statement,
        research_domains: determine_research_domains(idea)
    )
    
    // Parallel research execution across domains
    research_tasks = []
    FOR domain IN research_context.research_domains:
        task = create_research_task(research_context, domain)
        research_tasks.append(task)
    
    // TEST: should execute research tasks concurrently with timeout
    task_results = execute_parallel_with_timeout(
        tasks: research_tasks,
        timeout: get_config("RESEARCH_TIMEOUT") OR 300,  // 5 minutes
        max_concurrent: get_config("MAX_CONCURRENT_RESEARCH") OR 3
    )
    
    // Collect and validate results
    evidence_items = []
    failed_domains = []
    
    FOR domain, result IN task_results.items():
        IF result.is_success():
            validated_evidence = validate_domain_evidence(result.evidence, domain)
            evidence_items.extend(validated_evidence)
            update_progress(idea_id, domain, 1.0)
        ELSE:
            failed_domains.append(domain)
            logger.error("Research failed for domain", domain: domain, error: result.error)
            update_progress(idea_id, domain, 0.0)
    
    // Check minimum evidence requirements
    evidence_quality = assess_evidence_quality(evidence_items)
    IF NOT evidence_quality.meets_minimum_requirements():
        // TEST: should trigger retry for failed domains before failing
        retry_results = retry_failed_research(research_context, failed_domains)
        evidence_items.extend(retry_results.evidence)
    
    // Final validation and scoring
    overall_score = calculate_evidence_score(evidence_items)
    research_results = ResearchResults(
        idea_id: idea_id,
        evidence_items: evidence_items,
        overall_score: overall_score,
        research_domains: research_context.research_domains,
        failed_domains: failed_domains,
        completed_at: get_current_timestamp()
    )
    
    // Store results and update idea state
    save_research_results(research_results)
    advance_idea_stage_if_qualified(idea_id, research_results)
    
    RETURN Success(research_results)

FUNCTION determine_research_domains(idea: Idea) -> List[ResearchDomain]:
    // TEST: should include market analysis for all ideas
    // TEST: should include technical research for technology-focused ideas
    // TEST: should include regulatory research for healthcare/fintech ideas
    // TEST: should limit total domains to prevent excessive API costs
    
    domains = [ResearchDomain.MARKET_ANALYSIS]  // Always include market research
    
    // Domain-specific research based on idea content
    category_keywords = extract_keywords(idea.title + " " + idea.description)
    
    IF contains_technology_keywords(category_keywords):
        domains.append(ResearchDomain.TECHNICAL_FEASIBILITY)
    
    IF contains_competitive_keywords(category_keywords):
        domains.append(ResearchDomain.COMPETITIVE_LANDSCAPE)
    
    IF contains_regulatory_keywords(category_keywords):
        domains.append(ResearchDomain.REGULATORY_COMPLIANCE)
    
    // Limit to maximum domains to control costs
    max_domains = get_config("MAX_RESEARCH_DOMAINS") OR 4
    RETURN domains[:max_domains]
```

## 2. RAG-based Evidence Collector

### 2.1 Multi-Source Research Agent
```pseudocode
FUNCTION collect_evidence_for_domain(context: ResearchContext, domain: ResearchDomain) -> DomainEvidence:
    // TEST: should query multiple reliable sources for each domain
    // TEST: should extract relevant citations with proper attribution
    // TEST: should verify citation accessibility before including
    // TEST: should score evidence relevance and credibility
    
    search_queries = generate_domain_queries(context, domain)
    // TEST: should generate domain-specific search strategies
    
    evidence_items = []
    source_budget = get_config("MAX_SOURCES_PER_DOMAIN") OR 10
    
    FOR query IN search_queries:
        IF source_budget <= 0:
            BREAK
        
        try:
            search_results = execute_rag_search(query, domain)
            processed_results = process_search_results(search_results, context, domain)
            
            FOR result IN processed_results:
                IF source_budget <= 0:
                    BREAK
                
                evidence = create_evidence_item(result, context, domain)
                IF validate_evidence_item(evidence):
                    evidence_items.append(evidence)
                    source_budget -= 1
                    
        catch SearchError as e:
            logger.warn("Search failed for query", query: query, domain: domain, error: e)
            continue
    
    // Filter and rank evidence by relevance and credibility
    ranked_evidence = rank_evidence_by_quality(evidence_items, context)
    top_evidence = select_top_evidence(ranked_evidence, max_count: 5)
    
    domain_evidence = DomainEvidence(
        domain: domain,
        evidence_items: top_evidence,
        search_queries_used: search_queries,
        total_sources_found: evidence_items.length,
        average_credibility: calculate_average_credibility(top_evidence)
    )
    
    RETURN domain_evidence

FUNCTION execute_rag_search(query: String, domain: ResearchDomain) -> SearchResults:
    // TEST: should use appropriate search engines for different domains
    // TEST: should handle rate limiting with exponential backoff
    // TEST: should filter results by publication date and credibility
    // TEST: should extract structured data from unstructured sources
    
    search_config = get_domain_search_config(domain)
    search_engines = search_config.preferred_engines
    
    all_results = []
    
    FOR engine IN search_engines:
        try:
            engine_results = search_with_retry(
                engine: engine,
                query: query,
                max_results: search_config.max_results_per_engine,
                filters: search_config.filters
            )
            
            processed_results = extract_structured_data(engine_results)
            all_results.extend(processed_results)
            
        catch RateLimitError as e:
            logger.warn("Rate limit hit", engine: engine, query: query)
            apply_backoff_delay(engine, e.retry_after)
            continue
        catch SearchEngineError as e:
            logger.error("Search engine error", engine: engine, error: e)
            continue
    
    // Deduplicate and merge results
    deduplicated_results = remove_duplicate_sources(all_results)
    enriched_results = enrich_with_metadata(deduplicated_results)
    
    RETURN SearchResults(
        query: query,
        domain: domain,
        results: enriched_results,
        total_found: all_results.length,
        search_engines_used: search_engines
    )
```

## 3. Citation Verification Module

### 3.1 Accessibility and Credibility Checker
```pseudocode
FUNCTION verify_citation_accessibility(evidence: Evidence) -> VerificationResult:
    // TEST: should check if URLs are accessible and return valid content
    // TEST: should detect paywalls and subscription barriers
    // TEST: should validate that content matches claimed information
    // TEST: should handle redirects and expired links gracefully
    
    citation_url = evidence.citation_url
    
    // Basic URL validation
    IF NOT is_valid_url(citation_url):
        RETURN VerificationResult(
            verified: false,
            error: "Invalid URL format",
            accessibility_score: 0.0
        )
    
    // Check accessibility with retry logic
    access_result = check_url_accessibility_with_retry(citation_url)
    IF NOT access_result.is_accessible:
        RETURN VerificationResult(
            verified: false,
            error: access_result.error_message,
            accessibility_score: 0.0
        )
    
    // Content validation
    page_content = fetch_page_content(citation_url)
    content_analysis = analyze_page_content(page_content, evidence.claim_text)
    
    // Check for paywalls or access restrictions
    paywall_detected = detect_paywall_indicators(page_content)
    subscription_required = detect_subscription_requirements(page_content)
    
    accessibility_score = calculate_accessibility_score(
        is_accessible: access_result.is_accessible,
        has_paywall: paywall_detected,
        requires_subscription: subscription_required,
        content_relevance: content_analysis.relevance_score
    )
    
    RETURN VerificationResult(
        verified: access_result.is_accessible AND content_analysis.is_relevant,
        accessibility_score: accessibility_score,
        content_analysis: content_analysis,
        has_paywall: paywall_detected,
        requires_subscription: subscription_required
    )

FUNCTION assess_source_credibility(evidence: Evidence) -> CredibilityAssessment:
    // TEST: should evaluate source reputation based on domain authority
    // TEST: should check author credentials and publication quality
    // TEST: should assess content freshness and update frequency
    // TEST: should detect potential bias indicators in content
    
    source_domain = extract_domain(evidence.citation_url)
    
    // Domain authority assessment
    domain_authority = get_domain_authority_score(source_domain)
    domain_reputation = assess_domain_reputation(source_domain)
    
    // Content quality indicators
    content_quality = analyze_content_quality(evidence)
    author_credibility = assess_author_credibility(evidence.citation_source)
    publication_quality = assess_publication_standards(evidence)
    
    // Bias and objectivity assessment
    bias_analysis = detect_bias_indicators(evidence.claim_text, evidence.citation_title)
    objectivity_score = calculate_objectivity_score(evidence)
    
    // Temporal relevance
    publication_date = extract_publication_date(evidence)
    recency_score = calculate_recency_score(publication_date)
    
    // Composite credibility score
    credibility_score = calculate_weighted_credibility(
        domain_authority: domain_authority * 0.25,
        domain_reputation: domain_reputation * 0.20,
        content_quality: content_quality * 0.20,
        author_credibility: author_credibility * 0.15,
        objectivity_score: objectivity_score * 0.10,
        recency_score: recency_score * 0.10
    )
    
    RETURN CredibilityAssessment(
        overall_score: credibility_score,
        domain_authority: domain_authority,
        content_quality: content_quality,
        author_credibility: author_credibility,
        bias_indicators: bias_analysis,
        recency_score: recency_score,
        assessment_details: compile_assessment_details()
    )
```

## 4. Quality Scoring Engine

### 4.1 Evidence Quality Assessment
```pseudocode
FUNCTION calculate_evidence_score(evidence_items: List[Evidence]) -> EvidenceScore:
    // TEST: should require minimum number of evidence items (≥3)
    // TEST: should weight evidence by credibility and relevance
    // TEST: should penalize low-quality or inaccessible sources
    // TEST: should bonus for diverse, high-authority sources
    
    IF evidence_items.length < get_config("MIN_EVIDENCE_ITEMS") OR 3:
        RETURN EvidenceScore(
            overall_score: 0.0,
            meets_requirements: false,
            deficiency_reason: "Insufficient evidence items"
        )
    
    total_weighted_score = 0.0
    total_weight = 0.0
    quality_metrics = EvidenceQualityMetrics()
    
    FOR evidence IN evidence_items:
        // Individual evidence scoring
        relevance_score = calculate_relevance_score(evidence)
        credibility_score = evidence.credibility_score
        accessibility_score = evidence.accessibility_verified ? 1.0 : 0.5
        
        // Evidence item weight based on quality
        evidence_weight = calculate_evidence_weight(
            credibility: credibility_score,
            relevance: relevance_score,
            accessibility: accessibility_score
        )
        
        evidence_score = (relevance_score + credibility_score + accessibility_score) / 3.0
        total_weighted_score += evidence_score * evidence_weight
        total_weight += evidence_weight
        
        // Update quality metrics
        quality_metrics.add_evidence_metrics(evidence, evidence_score)
    
    // Calculate diversity bonus
    source_diversity = calculate_source_diversity(evidence_items)
    domain_coverage = calculate_domain_coverage(evidence_items)
    diversity_bonus = (source_diversity + domain_coverage) * 0.1
    
    // Final score calculation
    base_score = total_weighted_score / total_weight IF total_weight > 0 ELSE 0.0
    final_score = MIN(1.0, base_score + diversity_bonus)
    
    // Quality requirements check
    meets_requirements = check_quality_requirements(
        score: final_score,
        evidence_count: evidence_items.length,
        accessibility_rate: quality_metrics.accessibility_rate,
        credibility_average: quality_metrics.average_credibility
    )
    
    RETURN EvidenceScore(
        overall_score: final_score,
        meets_requirements: meets_requirements,
        evidence_count: evidence_items.length,
        average_credibility: quality_metrics.average_credibility,
        source_diversity: source_diversity,
        domain_coverage: domain_coverage,
        quality_metrics: quality_metrics
    )

FUNCTION check_quality_requirements(score: Float, evidence_count: Integer, 
                                   accessibility_rate: Float, credibility_average: Float) -> Boolean:
    // TEST: should enforce minimum score threshold
    // TEST: should require minimum accessibility rate
    // TEST: should require minimum average credibility
    // TEST: should be configurable via environment variables
    
    min_score = get_config("MIN_EVIDENCE_SCORE") OR 0.7
    min_accessibility = get_config("MIN_ACCESSIBILITY_RATE") OR 0.8
    min_credibility = get_config("MIN_CREDIBILITY_AVERAGE") OR 0.6
    
    RETURN score >= min_score AND 
           accessibility_rate >= min_accessibility AND 
           credibility_average >= min_credibility
```

## 5. Agent Coordination Module

### 5.1 Multi-Agent Research Orchestration
```pseudocode
FUNCTION coordinate_research_agents(idea_id: UUID, research_domains: List[ResearchDomain]) -> AgentResults:
    // TEST: should assign different agents to different research domains
    // TEST: should handle agent failures and reassign work
    // TEST: should merge results from multiple agents working on same domain
    // TEST: should enforce agent-specific timeouts and resource limits
    
    agent_assignments = assign_agents_to_domains(research_domains)
    active_tasks = {}
    completed_results = {}
    failed_assignments = []
    
    // Start agent tasks
    FOR domain, assigned_agent IN agent_assignments.items():
        task_context = create_agent_task_context(idea_id, domain)
        
        try:
            task_id = start_agent_task(assigned_agent, task_context)
            active_tasks[task_id] = AgentTaskInfo(
                agent: assigned_agent,
                domain: domain,
                started_at: get_current_timestamp(),
                timeout_at: get_current_timestamp() + get_agent_timeout(assigned_agent)
            )
        catch AgentStartupError as e:
            logger.error("Failed to start agent task", agent: assigned_agent, domain: domain, error: e)
            failed_assignments.append(domain)
    
    // Monitor and collect results
    WHILE active_tasks.length > 0:
        // Check for completed tasks
        completed_task_ids = check_completed_tasks(active_tasks.keys())
        
        FOR task_id IN completed_task_ids:
            task_info = active_tasks[task_id]
            
            try:
                result = collect_agent_result(task_id)
                completed_results[task_info.domain] = result
                logger.info("Agent task completed", agent: task_info.agent, domain: task_info.domain)
            catch ResultCollectionError as e:
                logger.error("Failed to collect agent result", task_id: task_id, error: e)
                failed_assignments.append(task_info.domain)
            
            active_tasks.remove(task_id)
        
        // Check for timeouts
        current_time = get_current_timestamp()
        timed_out_tasks = []
        
        FOR task_id, task_info IN active_tasks.items():
            IF current_time > task_info.timeout_at:
                timed_out_tasks.append(task_id)
        
        FOR task_id IN timed_out_tasks:
            task_info = active_tasks[task_id]
            cancel_agent_task(task_id)
            failed_assignments.append(task_info.domain)
            active_tasks.remove(task_id)
            logger.warn("Agent task timed out", agent: task_info.agent, domain: task_info.domain)
        
        sleep(1.0)  // Poll interval
    
    // Handle failed assignments with backup agents
    IF failed_assignments.length > 0:
        backup_results = handle_failed_assignments(idea_id, failed_assignments)
        completed_results.update(backup_results)
    
    RETURN AgentResults(
        completed_results: completed_results,
        failed_domains: failed_assignments,
        total_agents_used: count_unique_agents(agent_assignments, backup_results)
    )

FUNCTION assign_agents_to_domains(domains: List[ResearchDomain]) -> Dict[ResearchDomain, Agent]:
    // TEST: should select appropriate agents based on domain expertise
    // TEST: should balance load across available agents
    // TEST: should prefer agents with higher success rates
    // TEST: should handle cases where no agents are available for a domain
    
    assignments = {}
    available_agents = get_available_research_agents()
    
    FOR domain IN domains:
        suitable_agents = filter_agents_by_domain_expertise(available_agents, domain)
        
        IF suitable_agents.length == 0:
            logger.warn("No suitable agents for domain", domain: domain)
            continue
        
        // Select agent based on performance and availability
        selected_agent = select_best_agent(suitable_agents, domain)
        assignments[domain] = selected_agent
        
        // Update agent availability
        mark_agent_busy(selected_agent, estimated_duration: get_domain_work_estimate(domain))
    
    RETURN assignments
```

## 6. Research Results Storage

### 6.1 Evidence Persistence Layer
```pseudocode
FUNCTION save_research_results(results: ResearchResults) -> None:
    // TEST: should save all evidence items with proper relationships
    // TEST: should maintain audit trail of research activities
    // TEST: should handle concurrent saves for same idea
    // TEST: should validate data integrity before persistence
    
    BEGIN_TRANSACTION:
        try:
            // Save evidence items
            FOR evidence IN results.evidence_items:
                validated_evidence = validate_evidence_for_storage(evidence)
                evidence_repository.save(validated_evidence)
                
                // Update idea-evidence relationships
                link_evidence_to_idea(results.idea_id, evidence.evidence_id)
            
            // Update idea with research completion
            idea = idea_repository.find_by_id(results.idea_id)
            idea.evidence_score = results.overall_score
            idea.stage_progress = calculate_research_stage_progress(results)
            idea.updated_at = get_current_timestamp()
            
            idea_repository.save(idea)
            
            // Create audit entries
            audit_entry = ResearchAuditEntry(
                idea_id: results.idea_id,
                evidence_count: results.evidence_items.length,
                research_domains: results.research_domains,
                overall_score: results.overall_score,
                completed_at: results.completed_at
            )
            audit_repository.save(audit_entry)
            
            COMMIT_TRANSACTION
            
            // Publish research completion event
            publish_research_completed_event(results)
            
        catch StorageError as e:
            ROLLBACK_TRANSACTION
            logger.error("Failed to save research results", idea_id: results.idea_id, error: e)
            THROW e

FUNCTION validate_evidence_for_storage(evidence: Evidence) -> Evidence:
    // TEST: should ensure all required fields are present
    // TEST: should validate URL formats and accessibility
    // TEST: should check for duplicate evidence items
    // TEST: should sanitize text content for storage
    
    validation_errors = []
    
    IF evidence.claim_text IS EMPTY:
        validation_errors.append("Claim text is required")
    
    IF NOT is_valid_url(evidence.citation_url):
        validation_errors.append("Invalid citation URL")
    
    IF evidence.credibility_score < 0.0 OR evidence.credibility_score > 1.0:
        validation_errors.append("Credibility score must be between 0.0 and 1.0")
    
    // Check for duplicates
    existing_evidence = evidence_repository.find_by_url_and_claim(
        evidence.citation_url, 
        evidence.claim_text
    )
    IF existing_evidence IS NOT NULL:
        validation_errors.append("Duplicate evidence item")
    
    IF validation_errors.length > 0:
        THROW ValidationError("Evidence validation failed: " + validation_errors.join(", "))
    
    // Sanitize content
    sanitized_evidence = Evidence(
        evidence_id: evidence.evidence_id,
        idea_id: evidence.idea_id,
        claim_text: sanitize_text(evidence.claim_text),
        citation_url: evidence.citation_url,
        citation_title: sanitize_text(evidence.citation_title),
        citation_source: sanitize_text(evidence.citation_source),
        relevance_score: evidence.relevance_score,
        credibility_score: evidence.credibility_score,
        evidence_type: evidence.evidence_type,
        research_domain: evidence.research_domain,
        collected_by_agent: evidence.collected_by_agent,
        created_at: evidence.created_at
    )
    
    RETURN sanitized_evidence
```

## 7. Error Handling and Recovery

### 7.1 Research Failure Recovery
```pseudocode
FUNCTION handle_research_failures(idea_id: UUID, failed_domains: List[ResearchDomain]) -> RecoveryResults:
    // TEST: should attempt alternative research strategies for failed domains
    // TEST: should use backup data sources when primary sources fail
    // TEST: should gracefully degrade quality when some research fails
    // TEST: should not exceed maximum retry attempts or cost budgets
    
    recovery_results = RecoveryResults()
    retry_budget = get_config("MAX_RESEARCH_RETRIES") OR 2
    
    FOR domain IN failed_domains:
        retry_count = 0
        domain_recovered = false
        
        WHILE retry_count < retry_budget AND NOT domain_recovered:
            try:
                // Try alternative research approach
                alternative_strategy = get_alternative_research_strategy(domain, retry_count)
                recovery_evidence = execute_alternative_research(idea_id, domain, alternative_strategy)
                
                IF recovery_evidence.length > 0:
                    recovery_results.add_recovered_evidence(domain, recovery_evidence)
                    domain_recovered = true
                    logger.info("Research recovery successful", domain: domain, attempt: retry_count + 1)
                
            catch ResearchError as e:
                retry_count += 1
                logger.warn("Research recovery attempt failed", 
                          domain: domain, attempt: retry_count, error: e)
                
                IF retry_count < retry_budget:
                    backoff_delay = calculate_backoff_delay(retry_count)
                    sleep(backoff_delay)
        
        IF NOT domain_recovered:
            recovery_results.add_failed_domain(domain)
            logger.error("Research recovery exhausted for domain", domain: domain)
    
    RETURN recovery_results
```

---

**Summary**: This module orchestrates comprehensive evidence collection through multi-agent research, RAG-based information retrieval, citation verification, and quality assessment. All functions include extensive TDD anchors for thorough testing coverage and robust error handling for production reliability.