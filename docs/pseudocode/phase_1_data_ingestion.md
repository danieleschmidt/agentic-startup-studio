# Phase 1: Data Ingestion - Pseudocode Specification

## Module Overview
**Responsibility:** Capture, validate, and store startup ideas with structured metadata and similarity detection.
**Input:** Raw idea data from CLI, API, or batch import
**Output:** Validated Idea entity stored in database with unique identifier
**Quality Gates:** Input validation, duplication detection, schema compliance

## 1. CLI Interface Module

### 1.1 Idea Creation Command
```pseudocode
FUNCTION create_idea_cli(args: CommandLineArgs) -> Result[IdeaID, ValidationError]:
    // TEST: should accept valid idea with required fields
    // TEST: should reject empty title or description
    // TEST: should validate description length between 10-5000 characters
    // TEST: should set default status to DRAFT
    
    VALIDATE args.title IS NOT EMPTY AND LENGTH(args.title) BETWEEN 10 AND 200
    VALIDATE args.description IS NOT EMPTY AND LENGTH(args.description) BETWEEN 10 AND 5000
    VALIDATE args.category IN ALLOWED_CATEGORIES OR args.category IS NULL
    
    sanitized_data = sanitize_input_data(args)
    // TEST: should sanitize HTML and SQL injection attempts
    
    idea_draft = IdeaDraft(
        title: sanitized_data.title,
        description: sanitized_data.description,
        category: sanitized_data.category OR "uncategorized",
        problem_statement: sanitized_data.problem_statement,
        solution_description: sanitized_data.solution_description,
        target_market: sanitized_data.target_market
    )
    
    validation_result = validate_idea_draft(idea_draft)
    IF validation_result.has_errors():
        RETURN Error(validation_result.errors)
    
    duplicate_check = check_for_duplicates(idea_draft)
    IF duplicate_check.found_similar():
        // TEST: should prompt user for confirmation when similar ideas exist
        IF NOT args.force_create:
            RETURN Error("Similar ideas found: " + duplicate_check.similar_ideas)
    
    idea_id = save_new_idea(idea_draft)
    // TEST: should return valid UUID for successfully created idea
    RETURN Success(idea_id)

FUNCTION list_ideas_cli(filters: ListFilters) -> List[IdeaSummary]:
    // TEST: should return empty list when no ideas exist
    // TEST: should filter by status when status filter provided
    // TEST: should sort by created_at desc by default
    // TEST: should respect pagination limits
    
    query_params = build_query_from_filters(filters)
    ideas = idea_repository.find_with_filters(query_params)
    
    formatted_ideas = []
    FOR EACH idea IN ideas:
        summary = IdeaSummary(
            id: idea.idea_id,
            title: idea.title,
            status: idea.status,
            stage: idea.current_stage,
            created_at: idea.created_at,
            progress: calculate_progress_percentage(idea)
        )
        formatted_ideas.append(summary)
    
    RETURN formatted_ideas

FUNCTION update_idea_cli(idea_id: UUID, updates: UpdateData) -> Result[None, Error]:
    // TEST: should update existing idea with valid data  
    // TEST: should reject updates to non-existent ideas
    // TEST: should preserve audit trail of changes
    // TEST: should validate updated fields meet schema requirements
    
    existing_idea = idea_repository.find_by_id(idea_id)
    IF existing_idea IS NULL:
        RETURN Error("Idea not found: " + idea_id)
    
    IF existing_idea.status IN LOCKED_STATUSES:
        RETURN Error("Cannot modify idea in status: " + existing_idea.status)
    
    sanitized_updates = sanitize_input_data(updates)
    validation_result = validate_partial_update(existing_idea, sanitized_updates)
    IF validation_result.has_errors():
        RETURN Error(validation_result.errors)
    
    updated_idea = apply_updates(existing_idea, sanitized_updates)
    audit_entry = create_audit_entry("idea_updated", idea_id, updates, get_current_user())
    
    idea_repository.save(updated_idea)
    audit_repository.save(audit_entry)
    // TEST: should save audit entry with user attribution
    
    RETURN Success()
```

## 2. Input Validation Module

### 2.1 Core Validation Functions
```pseudocode
FUNCTION validate_idea_draft(draft: IdeaDraft) -> ValidationResult:
    // TEST: should validate all required fields are present
    // TEST: should check field length constraints
    // TEST: should validate enum values for category
    // TEST: should reject malicious input patterns
    
    errors = []
    
    IF draft.title IS EMPTY:
        errors.append("Title is required")
    ELSE IF LENGTH(draft.title) > 200:
        errors.append("Title must be under 200 characters")
    
    IF draft.description IS EMPTY:
        errors.append("Description is required")
    ELSE IF LENGTH(draft.description) < 10:
        errors.append("Description must be at least 10 characters")
    ELSE IF LENGTH(draft.description) > 5000:
        errors.append("Description must be under 5000 characters")
    
    IF draft.category NOT IN VALID_CATEGORIES:
        errors.append("Invalid category: " + draft.category)
    
    // Advanced validation rules
    profanity_check = scan_for_inappropriate_content(draft.title, draft.description)
    IF profanity_check.found_violations():
        errors.append("Content violates community guidelines")
    
    // TEST: should detect and flag potential spam patterns
    spam_check = analyze_for_spam_patterns(draft)
    IF spam_check.is_likely_spam():
        errors.append("Content appears to be spam")
    
    RETURN ValidationResult(errors: errors, is_valid: errors.is_empty())

FUNCTION sanitize_input_data(raw_data: Dict) -> SanitizedData:
    // TEST: should remove HTML tags from text fields
    // TEST: should escape SQL injection patterns
    // TEST: should normalize whitespace and line breaks
    // TEST: should preserve legitimate special characters
    
    sanitized = {}
    
    FOR field_name, field_value IN raw_data.items():
        IF field_name IN TEXT_FIELDS:
            sanitized[field_name] = html_escape(
                sql_escape(
                    normalize_whitespace(field_value)
                )
            )
        ELSE:
            sanitized[field_name] = field_value
    
    RETURN SanitizedData(sanitized)
```

## 3. Duplication Detection Module

### 3.1 Similarity Analysis
```pseudocode
FUNCTION check_for_duplicates(new_idea: IdeaDraft) -> DuplicateCheckResult:
    // TEST: should find exact title matches
    // TEST: should detect similar descriptions using vector similarity
    // TEST: should return similarity scores for ranking
    // TEST: should exclude archived ideas from similarity check
    
    // Exact title match check
    exact_matches = idea_repository.find_by_title_exact(new_idea.title)
    IF exact_matches.length > 0:
        RETURN DuplicateCheckResult(
            found_similar: true,
            exact_matches: exact_matches,
            similar_ideas: []
        )
    
    // Vector similarity search using pgvector
    description_embedding = generate_text_embedding(new_idea.description)
    // TEST: should use configurable similarity threshold (default 0.8)
    similarity_threshold = get_config("SIMILARITY_THRESHOLD") OR 0.8
    
    similar_ideas = idea_repository.find_similar_by_embedding(
        embedding: description_embedding,
        threshold: similarity_threshold,
        exclude_statuses: ["ARCHIVED", "REJECTED"]
    )
    
    // Additional fuzzy matching on title
    title_similar = idea_repository.find_similar_titles(
        title: new_idea.title,
        fuzzy_threshold: 0.7
    )
    
    all_similar = merge_and_deduplicate(similar_ideas, title_similar)
    ranked_similar = rank_by_similarity_score(all_similar, new_idea)
    
    RETURN DuplicateCheckResult(
        found_similar: ranked_similar.length > 0,
        exact_matches: [],
        similar_ideas: ranked_similar,
        similarity_scores: extract_scores(ranked_similar)
    )

FUNCTION generate_text_embedding(text: String) -> Vector:
    // TEST: should generate consistent embeddings for same text
    // TEST: should handle empty or very short text gracefully
    // TEST: should cache embeddings to avoid redundant API calls
    
    normalized_text = normalize_for_embedding(text)
    
    cached_embedding = embedding_cache.get(hash(normalized_text))
    IF cached_embedding IS NOT NULL:
        RETURN cached_embedding
    
    embedding = embedding_service.create_embedding(normalized_text)
    // TEST: should retry on API failures with exponential backoff
    
    embedding_cache.set(hash(normalized_text), embedding, TTL: 24_HOURS)
    RETURN embedding
```

## 4. Data Storage Module

### 4.1 Repository Operations
```pseudocode
FUNCTION save_new_idea(draft: IdeaDraft) -> UUID:
    // TEST: should generate unique UUID for new ideas
    // TEST: should set created_at and updated_at timestamps
    // TEST: should initialize workflow state to IDEATE stage
    // TEST: should handle database connection failures gracefully
    
    idea_id = generate_uuid()
    current_time = get_current_timestamp()
    
    idea = Idea(
        idea_id: idea_id,
        title: draft.title,
        description: draft.description,
        category: draft.category,
        status: IdeaStatus.DRAFT,
        current_stage: PipelineStage.IDEATE,
        stage_progress: 0.0,
        created_at: current_time,
        updated_at: current_time,
        problem_statement: draft.problem_statement,
        solution_description: draft.solution_description,
        target_market: draft.target_market
    )
    
    // Generate and store embedding for similarity search
    description_embedding = generate_text_embedding(draft.description)
    
    BEGIN_TRANSACTION:
        try:
            idea_repository.save(idea)
            embedding_repository.save_idea_embedding(idea_id, description_embedding)
            audit_repository.save(create_audit_entry("idea_created", idea_id, draft))
            COMMIT_TRANSACTION
            // TEST: should emit IdeaCreated domain event
            event_publisher.publish(IdeaCreated(idea_id, draft.title, current_time))
        catch DatabaseError as e:
            ROLLBACK_TRANSACTION
            // TEST: should log error details for debugging
            logger.error("Failed to save new idea", error: e, draft: draft)
            THROW StorageError("Failed to save idea: " + e.message)
    
    RETURN idea_id

FUNCTION find_ideas_with_filters(query_params: QueryParams) -> List[Idea]:
    // TEST: should apply status filters correctly
    // TEST: should respect pagination limits and offsets
    // TEST: should sort results by specified field and direction
    // TEST: should return empty list for no matches
    
    sql_query = build_sql_from_params(query_params)
    // TEST: should prevent SQL injection through parameterized queries
    
    try:
        results = database.execute_query(sql_query, query_params.values)
        ideas = map_results_to_entities(results)
        
        // Apply post-query filters that can't be done in SQL
        IF query_params.has_similarity_filter():
            ideas = filter_by_similarity(ideas, query_params.similarity_criteria)
        
        RETURN ideas
    catch DatabaseError as e:
        logger.error("Query failed", query: sql_query, params: query_params, error: e)
        THROW QueryError("Failed to retrieve ideas: " + e.message)
```

## 5. Configuration Management

### 5.1 Environment Configuration
```pseudocode
FUNCTION load_ingestion_config() -> IngestionConfig:
    // TEST: should load config from environment variables
    // TEST: should use default values when env vars not set
    // TEST: should validate config values are within acceptable ranges
    // TEST: should fail fast on invalid configuration
    
    config = IngestionConfig(
        similarity_threshold: get_env_float("SIMILARITY_THRESHOLD") OR 0.8,
        max_description_length: get_env_int("MAX_DESCRIPTION_LENGTH") OR 5000,
        min_description_length: get_env_int("MIN_DESCRIPTION_LENGTH") OR 10,
        enable_spam_detection: get_env_bool("ENABLE_SPAM_DETECTION") OR true,
        enable_profanity_filter: get_env_bool("ENABLE_PROFANITY_FILTER") OR true,
        embedding_cache_ttl: get_env_int("EMBEDDING_CACHE_TTL") OR 86400,
        database_timeout: get_env_int("DATABASE_TIMEOUT") OR 30
    )
    
    validation_errors = validate_config(config)
    IF validation_errors.length > 0:
        THROW ConfigurationError("Invalid configuration: " + validation_errors.join(", "))
    
    RETURN config

FUNCTION validate_config(config: IngestionConfig) -> List[String]:
    // TEST: should validate numeric ranges are sensible
    // TEST: should check boolean values are properly set
    // TEST: should validate dependent configuration combinations
    
    errors = []
    
    IF config.similarity_threshold < 0.0 OR config.similarity_threshold > 1.0:
        errors.append("similarity_threshold must be between 0.0 and 1.0")
    
    IF config.max_description_length <= config.min_description_length:
        errors.append("max_description_length must be greater than min_description_length")
    
    IF config.database_timeout <= 0:
        errors.append("database_timeout must be positive")
    
    RETURN errors
```

## 6. Error Handling and Resilience

### 6.1 Retry Logic and Circuit Breakers
```pseudocode
FUNCTION with_retry(operation: Function, max_attempts: Integer) -> Result:
    // TEST: should retry failed operations up to max_attempts
    // TEST: should use exponential backoff between retries
    // TEST: should not retry on validation errors (non-transient)
    // TEST: should log all retry attempts
    
    attempt = 1
    base_delay = 1.0  // seconds
    
    WHILE attempt <= max_attempts:
        try:
            result = operation()
            IF attempt > 1:
                logger.info("Operation succeeded after retry", attempt: attempt)
            RETURN Success(result)
        catch TransientError as e:
            IF attempt == max_attempts:
                logger.error("Operation failed after all retries", attempts: attempt, error: e)
                THROW e
            
            delay = base_delay * (2 ** (attempt - 1))  // exponential backoff
            logger.warn("Operation failed, retrying", attempt: attempt, delay: delay, error: e)
            sleep(delay)
            attempt += 1
        catch NonTransientError as e:
            // Don't retry validation errors, permission errors, etc.
            logger.error("Operation failed with non-transient error", error: e)
            THROW e
    
    THROW MaxRetriesExceededError("Operation failed after " + max_attempts + " attempts")

FUNCTION database_operation_with_circuit_breaker(operation: Function) -> Result:
    // TEST: should open circuit after consecutive failures
    // TEST: should allow operations when circuit is closed
    // TEST: should transition to half-open state after timeout
    // TEST: should close circuit after successful operation in half-open state
    
    circuit_state = circuit_breaker.get_state()
    
    IF circuit_state == CircuitState.OPEN:
        IF circuit_breaker.should_attempt_reset():
            circuit_breaker.set_state(CircuitState.HALF_OPEN)
        ELSE:
            THROW CircuitOpenError("Database circuit breaker is open")
    
    try:
        result = operation()
        IF circuit_state == CircuitState.HALF_OPEN:
            circuit_breaker.set_state(CircuitState.CLOSED)
            logger.info("Circuit breaker reset to closed state")
        RETURN result
    catch DatabaseError as e:
        circuit_breaker.record_failure()
        IF circuit_breaker.should_open():
            circuit_breaker.set_state(CircuitState.OPEN)
            logger.error("Circuit breaker opened due to failures")
        THROW e
```

## 7. Integration Points

### 7.1 Event Publishing
```pseudocode
FUNCTION publish_idea_events(idea: Idea, event_type: String) -> None:
    // TEST: should publish events to configured message broker
    // TEST: should include all relevant idea metadata in events
    // TEST: should handle publishing failures gracefully
    // TEST: should not block main operation if publishing fails
    
    event_data = IdeaEvent(
        event_type: event_type,
        idea_id: idea.idea_id,
        title: idea.title,
        status: idea.status,
        stage: idea.current_stage,
        timestamp: get_current_timestamp(),
        metadata: extract_event_metadata(idea)
    )
    
    try:
        event_publisher.publish_async(event_data)
        logger.debug("Published idea event", event_type: event_type, idea_id: idea.idea_id)
    catch PublishingError as e:
        // Don't fail the main operation, but log the error
        logger.error("Failed to publish idea event", error: e, event: event_data)
        // Could queue for retry later
        failed_events_queue.enqueue(event_data)
```

## 8. Performance Monitoring

### 8.1 Metrics Collection
```pseudocode
FUNCTION collect_ingestion_metrics(operation: String, duration: Float, success: Boolean) -> None:
    // TEST: should record operation timing metrics
    // TEST: should track success/failure rates
    // TEST: should increment counters for different operation types
    
    metrics.timer("idea_ingestion.operation_duration", duration, tags: [operation])
    metrics.counter("idea_ingestion.operations_total", tags: [operation, success_status(success)])
    
    IF NOT success:
        metrics.counter("idea_ingestion.errors_total", tags: [operation])
    
    // Track data quality metrics
    IF operation == "create_idea":
        metrics.gauge("idea_ingestion.queue_size", get_pending_ideas_count())
```

---

**Summary**: This module handles the complete data ingestion workflow with comprehensive validation, duplication detection, secure storage, and robust error handling. All functions include TDD anchors for thorough testing coverage.